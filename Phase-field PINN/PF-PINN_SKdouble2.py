from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
import scipy.io
import time
import pandas as pd

# 定义子网络列表
def NN_list(layers):
    depth = len(layers) - 1
    activation = torch.nn.SiLU
    layer_list = list()
    for i in range(depth - 1):
        linear_layer = torch.nn.Linear(layers[i], layers[i + 1])
        # 手动进行 Xavier 初始化
        torch.nn.init.xavier_uniform_(linear_layer.weight)
        layer_list.append(('layer_%d' % i, linear_layer))
        layer_list.append(('activation_%d' % i, activation()))
    # 最后一层不需要激活函数
    linear_layer = torch.nn.Linear(layers[-2], layers[-1])
    torch.nn.init.xavier_uniform_(linear_layer.weight)
    layer_list.append(('layer_%d' % (depth - 1), linear_layer))
    layerDict = OrderedDict(layer_list)
    return layerDict

# 定义网络
class DNN(torch.nn.Module):
    def __init__(self, layers,layers_u,layers_v,layers_p,layers_C):
        super(DNN, self).__init__()
        self.layers = torch.nn.Sequential(NN_list(layers)).float()
        self.layers_u = torch.nn.Sequential(NN_list(layers_u)).float()
        self.layers_v = torch.nn.Sequential(NN_list(layers_v)).float()
        self.layers_p = torch.nn.Sequential(NN_list(layers_p)).float()
        self.layers_C = torch.nn.Sequential(NN_list(layers_C)).float()

    def forward(self, x):
        out = self.layers(x)
        out = torch.cat([self.layers_u(out),self.layers_v(out),self.layers_p(out),self.layers_C(out)], dim=1)
        return out

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PhysicsInformedNN():
    # 初始化
    def __init__(self, xt_area, neighborhoods, distances, distance_vectors,h,order,
                 x_ic,u_ic,x_bc1,x_bc2, dxdy,B, layers,layers_u,layers_v,layers_p,layers_C,Adam_iter):
        self.xt_area = torch.tensor(xt_area).float().to(device)
        self.neighborhoods = neighborhoods.to(device)
        self.distances = distances.float().to(device)
        self.distance_vectors = distance_vectors.float().to(device)
        self.kernel = sph_kernel(distances, h).float().to(device)
        self.RKPM_C = Compute_C(self.distance_vectors, self.kernel.unsqueeze(-1), dxdy,order).float().to(device)
        self.x_bc1 = torch.tensor(x_bc1).float().to(device)
        self.x_bc2 = torch.tensor(x_bc2, requires_grad=True).float().to(device)
        self.x_ic = torch.tensor(x_ic).float().to(device)
        self.u_ic = torch.tensor(u_ic).float().to(device)
        self.dxdy = dxdy
        # 定义一个深度网络
        self.B = torch.tensor(B).float().to(device)
        self.dnn = DNN(layers,layers_u,layers_v,layers_p,layers_C).to(device)

        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), 0.001)
        self.Adam_iter = Adam_iter
        self.iter = 0
        self.loss = []

    def net_u(self, xyt):
        X = torch.cat([torch.cos(xyt @ self.B), torch.sin(xyt @ self.B)], dim=1)
        u = self.dnn(X)
        return u

    def net_f(self, xyt, i_batch, j_batch):
        batch_neighborhoods = self.neighborhoods[i_batch]
        batch_neighbor_xt_area = self.xt_area[batch_neighborhoods][torch.arange(j_batch.shape[0]), :, j_batch, :] * (batch_neighborhoods!= -1).unsqueeze(-1)
        batch_neighbor_kernel = self.kernel[i_batch].unsqueeze(-1)
        batch_neighbor_RKPM_C = self.RKPM_C[i_batch]
        uvpc = self.net_u(xyt)
        uvpc_neighbor = self.net_u(batch_neighbor_xt_area.reshape(-1,3)).reshape(j_batch.shape[0],-1,4)
        C_neighbor = uvpc_neighbor[:,:, 3:4]*(torch.abs(uvpc_neighbor[:,:, 3:4]) <= 1) + torch.sign(uvpc_neighbor[:,:, 3:4])*(torch.abs(uvpc_neighbor[:,:, 3:4]) > 1)
        C = uvpc[:, 3:4]*(torch.abs(uvpc[:, 3:4]) <= 1) + torch.sign(uvpc[:, 3:4])*(torch.abs(uvpc[:, 3:4]) > 1)
        rho = 0.5 * (1 + C) * 1000.0 + 0.5 * (1 - C) * 1.0
        mu = 0.5 * (1 + C) * 10.0 + 0.5 * (1 - C) * 0.1
        u_d = R_compute_field_gradients(uvpc_neighbor[:,:, 0:1], batch_neighbor_kernel, batch_neighbor_RKPM_C)
        v_d = R_compute_field_gradients(uvpc_neighbor[:,:, 1:2], batch_neighbor_kernel, batch_neighbor_RKPM_C)
        p_d = R_compute_field_gradients(uvpc_neighbor[:,:, 2:3], batch_neighbor_kernel, batch_neighbor_RKPM_C)
        c_d = R_compute_field_gradients(C_neighbor, batch_neighbor_kernel, batch_neighbor_RKPM_C)
        u_t = torch.autograd.grad(uvpc[:, 0:1], xyt, grad_outputs=torch.ones_like(uvpc[:, 0:1]), retain_graph=True, create_graph=True)[0][:,2:3]
        v_t = torch.autograd.grad(uvpc[:, 1:2], xyt, grad_outputs=torch.ones_like(uvpc[:, 1:2]), retain_graph=True, create_graph=True)[0][:,2:3]
        c_t = torch.autograd.grad(uvpc[:, 3:4], xyt, grad_outputs=torch.ones_like(uvpc[:, 3:4]), retain_graph=True, create_graph=True)[0][:,2:3]
        fai_d = self.fai_d(j_batch,C_neighbor,batch_neighborhoods,batch_neighbor_kernel, batch_neighbor_RKPM_C)
        fai = C*(C**2-1)-0.01**2 * (c_d[:,2:3]+c_d[:,4:5])
        fsigma = (3 * 2 ** .5 / 4) * (1.96 / 0.01) * fai * c_d[:, 0:2]

        e1 = u_d[:, 0:1] + v_d[:, 1:2]
        e2 = (rho * (u_t+uvpc[:, 0:1] * u_d[:, 0:1] + uvpc[:, 1:2] * u_d[:, 1:2]) \
             + 1 *(p_d[:, 0:1]-0.5 * 9.9 * c_d[:,1:2]*(u_d[:,1:2]+v_d[:,0:1])-9.9*c_d[:,0:1]*u_d[:,0:1]-mu*(u_d[:, 2:3] + u_d[:, 4:5])-fsigma[:,0:1]) - 0.0)/1000.0
        e3 = (rho * (v_t+uvpc[:, 0:1] * v_d[:, 0:1] + uvpc[:, 1:2] * v_d[:, 1:2]) \
             + 1 *(p_d[:, 1:2]-0.5 * 9.9 * c_d[:,0:1]*(u_d[:,1:2]+v_d[:,0:1])-9.9*c_d[:,1:2]*v_d[:,1:2]-mu*(v_d[:, 2:3] + v_d[:, 4:5])-fsigma[:,1:2]) - rho * 0.98)/1000.0
        e4 = c_t + uvpc[:, 0:1]*c_d[:,0:1] + uvpc[:, 1:2]*c_d[:,1:2] - 1e-4*(fai_d[:,2:3]+fai_d[:,4:5])
        return e1,e2,e3,e4

    def fai_d(self,j_batch,C_neighbor,batch_neighborhoods,batch_neighbor_kernel, batch_neighbor_RKPM_C):
        _, N = batch_neighborhoods.size()
        batch_neighborhoods2 = self.neighborhoods[batch_neighborhoods]
        batch_neighbor_xt_area2 = self.xt_area[batch_neighborhoods2][torch.arange(j_batch.shape[0]), :, :, j_batch, :] * (batch_neighborhoods2 != -1).unsqueeze(-1)
        C_neighbor2 = self.net_u(batch_neighbor_xt_area2.reshape(-1, 3))[:, 3].reshape(j_batch.shape[0], N, N)
        C_neighbor2 = C_neighbor2 * (torch.abs(C_neighbor2) <= 1) + torch.sign(C_neighbor2) * (torch.abs(C_neighbor2) > 1)
        batch_neighbor_kernel2 = self.kernel[batch_neighborhoods] * (batch_neighborhoods2 != -1)
        batch_neighbor_RKPM_C2 = self.RKPM_C[batch_neighborhoods] * (batch_neighborhoods2 != -1).unsqueeze(-1)
        c_d = R_compute_field_gradients(C_neighbor2.unsqueeze(-1), batch_neighbor_kernel2.unsqueeze(-1), batch_neighbor_RKPM_C2)
        fai = C_neighbor*(C_neighbor**2-1) - 1e-4 * (c_d[:,:,2:3]+c_d[:,:,4:5])
        fai_d = R_compute_field_gradients(fai, batch_neighbor_kernel, batch_neighbor_RKPM_C)
        return fai_d

    def net_bc2(self, xyt):
        uvpc = self.net_u(xyt)
        v_d = torch.autograd.grad(uvpc[:, 1:2], xyt, grad_outputs=torch.ones_like(uvpc[:, 1:2]), retain_graph=True, create_graph=True)[0]
        return uvpc[:, 0:1], v_d[:, 0:1]

    def Calculate_loss(self,x_ic_batch, u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch):
        ic = self.net_u(x_ic_batch)
        bc1 = self.net_u(x_bc1_batch)
        bc2u,bc2dvdx = self.net_bc2(x_bc2_batch)
        f_e1,f_e2,f_e3,f_e4 = self.net_f(xt_area_batch, i_batch, j_batch)
        loss_icC = torch.mean((ic[:,3:4]-u_ic_batch) ** 2)
        loss_icuv = torch.mean((ic[:, 0:2]) ** 2)
        loss_bc1 = torch.mean((bc1[:,0:2]) ** 2)
        loss_bc2u = torch.mean((bc2u) ** 2)
        loss_bc2dvdx = torch.mean((bc2dvdx) ** 2)
        loss_fe1 = torch.mean(f_e1 ** 2)
        loss_fe2 = torch.mean(f_e2 ** 2)
        loss_fe3 = torch.mean(f_e3 ** 2)
        loss_fe4 = torch.mean(f_e4 ** 2)
        loss = 100.0 * loss_icC + 1.0 * loss_icuv + 1.0 * loss_bc1 + 1.0 * loss_bc2u + 1.0 * loss_bc2dvdx + 1.0 * loss_fe1 + 1.0 * loss_fe2 + 1.0 * loss_fe3 + 1.0 * loss_fe4

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, loss_icC: %.5e, loss_icuv: %.5e, loss_bc1: %.5e, loss_bc2u: %.5e, loss_bc2dvdx: %.5e, loss_fe1: %.5e, Loss_fe2: %.5e, Loss_fe3: %.5e, Loss_fe4: %.5e\n'
                #
                % (self.iter, loss.item(), loss_icC.item(), loss_icuv.item(), loss_bc1.item(), loss_bc2u.item(), loss_bc2dvdx.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item(), loss_fe4.item())
                # )
            )
        self.loss.append([self.iter, loss.item(), loss_icC.item(), loss_icuv.item(), loss_bc1.item(), loss_bc2u.item(), loss_bc2dvdx.item()])
        return loss

    def get_batch(self,batch_size_ic=150, batch_size_bc1=45,
                  batch_size_bc2=45, batch_size_area=1500):
        """
        从输入数据中划分批次数据集。
        参数:
            x_ic: 初始条件数据，形状 (80601, 3)
            x_bc1: 边界条件数据1，形状 (40602, 3)
            x_bc2: 边界条件数据2，形状 (81002, 3)
            xt_area: 区域数据，形状 (80601, 101, 3)
            batch_size_ic: x_ic 的批次大小，默认为 2000
            batch_size_bc1: x_bc1 的批次大小，默认为 600
            batch_size_bc2: x_bc2 的批次大小，默认为 1200
            batch_size_area: xt_area 的批次大小，默认为 20000

        返回:
            x_ic_batch: 形状 (2000, 3)
            x_bc1_batch: 形状 (600, 3)
            x_bc2_batch: 形状 (1200, 3)
            xt_area_batch: 形状 (20000, 3)
            i_batch: xt_area_batch 对应的 i 索引，形状 (20000,)
            j_batch: xt_area_batch 对应的 j 索引，形状 (20000,)
        """
        # 1. 为 x_ic 采样
        ic_indices = np.random.choice(self.x_ic.shape[0], size=batch_size_ic, replace=False)
        x_ic_batch = self.x_ic[ic_indices, :]
        u_ic_batch = self.u_ic[ic_indices, :]
        # 2. 为 x_bc1 采样
        bc1_indices = np.random.choice(self.x_bc1.shape[0], size=batch_size_bc1, replace=False)
        x_bc1_batch = self.x_bc1[bc1_indices, :]
        # 3. 为 x_bc2 采样
        bc2_indices = np.random.choice(self.x_bc2.shape[0], size=batch_size_bc2, replace=False)
        x_bc2_batch = self.x_bc2[bc2_indices, :]
        # 4. 为 xt_area 采样
        n_i, n_j, _ = self.xt_area.shape  # n_i = 80601, n_j = 101
        total_points = n_i * n_j  # 总点数 80601 * 101
        k_batch = np.random.choice(total_points, size=batch_size_area, replace=False)
        i_batch = k_batch // n_j  # 计算 i 索引
        j_batch = k_batch % n_j  # 计算 j 索引
        xt_area_batch = self.xt_area[i_batch, j_batch, :]
        return x_ic_batch,u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch

    def train(self):
        self.dnn.train()
        # 使用Adam训练
        for i in range(1,self.Adam_iter):
            x_ic_batch, u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch = self.get_batch()
            self.optimizer_adam.zero_grad()
            xt_area_batch.requires_grad_(True)
            loss = self.Calculate_loss(x_ic_batch, u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch)
            loss.backward()
            self.optimizer_adam.step()

    def predict(self, xyt):
        xyt = torch.tensor(xyt).float().to(device)
        self.dnn.eval()
        u = self.net_u(xyt)[:, 0:1]
        v = self.net_u(xyt)[:, 1:2]
        c = self.net_u(xyt)[:, 3:4]
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        return u,v,c

# 领域点搜索
def find_neighborhood_points(coords, radius):
    # 创建一个 cKDTree
    kdtree = cKDTree(coords)
    # 查询每个点的邻域点
    neighborhoods = []
    distances = []
    distance_vectors = []
    for i, point in enumerate(coords):
        # 查询在指定半径内的邻域点
        neighbors_indices = kdtree.query_ball_point(point, radius)
        # 获取邻域点的坐标
        neighbors_coords = coords[neighbors_indices]
        # 计算邻域点与当前点的距离
        dist = np.linalg.norm(neighbors_coords - point,axis=1)
        # 计算距离向量
        dist_vector = neighbors_coords - point
        neighborhoods.append(torch.tensor(neighbors_indices,dtype=torch.long))
        distances.append(torch.tensor(dist,dtype=torch.double))
        distance_vectors.append(torch.tensor(dist_vector,dtype=torch.double))
    neighborhoods = pad_sequence(neighborhoods, batch_first=True, padding_value=-1)
    distances = pad_sequence(distances, batch_first=True, padding_value=-1)
    distance_vectors = pad_sequence(distance_vectors, batch_first=True, padding_value=0)
    return neighborhoods, distances, distance_vectors


def sph_kernel(distances, h):
    q = distances / h
    result = torch.zeros_like(distances, dtype=torch.float64)
    # 计算核函数的值
    within_range = (0 <= q) & (q <= 2)
    result[within_range] = (15 / (7 * np.pi * h ** 2)) * (
            (2/3 - 1.0 * q[within_range] ** 2 + 0.5 * q[within_range] ** 3) * (q[within_range] <= 1) +
            (1/6 * (2 - q[within_range]) ** 3) * ((1 < q[within_range]) & (q[within_range] <= 2))
    )
    return result

# 再生核粒子方法
def Compute_C(distance_vectors, kernel, dxdy, order):
    moment_terms = [torch.ones(kernel.shape).to(device)]
    terms_num = np.sum(np.arange(1, order + 2))
    for i in range(1, order + 1):
        for j in range(i + 1):
            term = (distance_vectors[:, :, 0:1] ** (i - j)) * (distance_vectors[:, :, 1:2] ** j) / (dxdy ** (i / 2))
            moment_terms.append(term)
    moment_vector = torch.cat(moment_terms, dim=2)
    H = torch.tensor([[0, 1 / dxdy ** 0.5, 0, 0, 0, 0],
                       [0, 0, 1 / dxdy ** 0.5, 0, 0, 0],
                       [0, 0, 0, 2 / dxdy, 0, 0],
                       [0, 0, 0, 0, 1 / dxdy, 0],
                       [0, 0, 0, 0, 0, 2 / dxdy]], dtype=torch.float)
    H0 = torch.nn.functional.pad(H, (0, terms_num-H.shape[1]), value=0).to(device)
    matrix = torch.matmul(moment_vector.unsqueeze(3), moment_vector.unsqueeze(2)) * kernel.unsqueeze(-1)
    matrix_sum = torch.sum(matrix, dim=1)
    matrix_inverse = torch.inverse(matrix_sum)
    C = torch.matmul(torch.matmul(moment_vector, matrix_inverse), H0.t().view(1, 1, terms_num, -1)).squeeze(0)
    return C

def R_compute_field_gradients(f_values,kernel,C):
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(f_values*C*kernel,dim = -2)
    return field_gradients

dx = dy = 0.005
dxdy = dx*dy
dt = 0.005

x = np.arange(-0.5+0.0*dx,0.5+1.0*dx,dx)
y = np.arange(-1.0+0.0*dy,1.0+1.0*dy,dy)
t = np.arange(0.0*dt,0.5+1.0*dt,dt)
X, Y, T = np.meshgrid(x, y, t)

X_flat = X.flatten()
Y_flat = Y.flatten()
T_flat = T.flatten()

X_initial = X_flat[T_flat == 0]
Y_initial = Y_flat[T_flat == 0]
T_initial = T_flat[T_flat == 0]

X_F = X_flat[T_flat == 0.5]
Y_F = Y_flat[T_flat == 0.5]
T_F = T_flat[T_flat == 0.5]

xB = np.array([-0.5, 0.5])
X, Y, T = np.meshgrid(xB, y, t)

XB1_flat = X.flatten()
YB1_flat = Y.flatten()
TB1_flat = T.flatten()

YB = np.array([-1.0, 1.0])
X, Y, T = np.meshgrid(x, YB, t)

XB2_flat = X.flatten()
YB2_flat = Y.flatten()
TB2_flat = T.flatten()


xt_area = np.column_stack((X_flat, Y_flat, T_flat))
xt_area = xt_area.reshape(-1, len(t), 3)
x_area = xt_area[:,0,0:2]
x_ic = np.column_stack((X_initial, Y_initial, T_initial))
u_ic = -np.tanh(((x_ic[:,0:1]**2+(x_ic[:,1:2]+0.5)**2)**.5-0.15)/(2**.5*0.01))
x_bc1 = np.column_stack((XB2_flat, YB2_flat, TB2_flat))
x_bc2 = np.column_stack((XB1_flat, YB1_flat, TB1_flat))
x_f = np.column_stack((X_F, Y_F, T_F))
# # 创建散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(x_ic[:,0], x_ic[:,1], c=u_ic, cmap='jet')
# plt.colorbar(label='Density')  # 添加颜色条
# plt.title('2D Scatter Cloud Plot')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()
N1 = 3
N2 = 64
B = np.random.normal(0, 2, (N1,N2))
layers = [2*N2]+[64]*4
layers_u = [64]*4+[1]
layers_v = [64]*4+[1]
layers_p = [64]*4+[1]
layers_C = [64]*4+[1]

h = dx*1.4
neighborhoods, distances, distance_vectors = find_neighborhood_points(x_area, 2.0*h)
order = 2
Adam_iter = 100
model = PhysicsInformedNN(xt_area, neighborhoods, distances, distance_vectors,h,order,
                 x_ic,u_ic,x_bc1,x_bc2, dxdy,B, layers,layers_u,layers_v,layers_p,layers_C,Adam_iter)

# model.dnn.load_state_dict(torch.load("modelDNNSK.pth"))
# print("网络参数已加载！")
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')

# torch.save(model.dnn.state_dict(), "modelDNNSK.pth")
# print("网络参数已保存！")

x = np.arange(-0.5+0.0*dx,0.5+1.0*dx,10*dx)
y = np.arange(-1.0+0.0*dy,1.0+1.0*dy,10*dy)
t = np.arange(0.0*dt, 0.5+1.0*dt, 10*dt)
X, Y, T = np.meshgrid(x, y, t)

X_flat = X.flatten()
Y_flat = Y.flatten()
T_flat = T.flatten()
xt_area = np.column_stack((X_flat, Y_flat, T_flat))
u_pred,v_pred,c_pred = model.predict(xt_area)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xt_area[:,0], xt_area[:,1], xt_area[:,2], c=c_pred, cmap='jet', marker='o')
cbar = fig.colorbar(scatter, ax=ax)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xt_area[:,0], xt_area[:,1], xt_area[:,2], c=u_pred, cmap='jet', marker='o')
cbar = fig.colorbar(scatter, ax=ax)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(xt_area[:,0], xt_area[:,1], xt_area[:,2], c=v_pred, cmap='jet', marker='o')
cbar = fig.colorbar(scatter, ax=ax)

# 创建散点图
_,_,cic_pred = model.predict(x_ic)
plt.figure(figsize=(6, 8))
plt.scatter(x_ic[:,0], x_ic[:,1], c = cic_pred, cmap='jet')
plt.colorbar(label='Density')  # 添加颜色条
plt.title('2D Scatter Cloud Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.axis('equal')
plt.show()

uic_pred,vic_pred,cic_pred = model.predict(x_f)
plt.figure(figsize=(6, 8))
plt.scatter(x_ic[:,0], x_ic[:,1], c = cic_pred, cmap='jet')
plt.colorbar(label='Density')  # 添加颜色条
plt.title('2D Scatter Cloud Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 8))
plt.scatter(x_f[:,0], x_f[:,1], c = uic_pred, cmap='jet')
plt.colorbar(label='Density')  # 添加颜色条
plt.title('2D Scatter Cloud Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.axis('equal')
plt.show()

plt.figure(figsize=(6, 8))
plt.scatter(x_f[:,0], x_f[:,1], c = vic_pred, cmap='jet')
plt.colorbar(label='Density')  # 添加颜色条
plt.title('2D Scatter Cloud Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.axis('equal')
plt.show()