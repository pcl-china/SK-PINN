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
    activation = torch.nn.Tanh
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
    def __init__(self, layers,layers_u,layers_v,layers_p):
        super(DNN, self).__init__()
        self.layers = torch.nn.Sequential(NN_list(layers)).double()
        self.layers_u = torch.nn.Sequential(NN_list(layers_u)).double()
        self.layers_v = torch.nn.Sequential(NN_list(layers_v)).double()
        self.layers_p = torch.nn.Sequential(NN_list(layers_p)).double()

    def forward(self, x):
        out = self.layers(x)
        out = torch.cat([self.layers_u(out),self.layers_v(out) ,self.layers_p(out)], dim=1)
        return out

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PhysicsInformedNN():
    # 初始化
    def __init__(self,AD_points,SK_points, neighborhoods, distances, distance_vectors,h,Re,B,order,
                 x_bc,U_bc, dxdy, layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter):
        self.x_AD = torch.tensor(AD_points[:, 0:1], requires_grad=True).double().to(device)
        self.y_AD = torch.tensor(AD_points[:, 1:2], requires_grad=True).double().to(device)
        self.x_SK = torch.tensor(SK_points[:, 0:1], requires_grad=True).double().to(device)
        self.y_SK = torch.tensor(SK_points[:, 1:2], requires_grad=True).double().to(device)
        self.neighborhoods = neighborhoods.to(device)
        self.distances = distances.to(device)
        self.distance_vectors = distance_vectors.to(device)
        self.kernel = sph_kernel(distances, h).to(device)
        self.RKPM_C = Compute_C(self.distance_vectors, self.kernel.unsqueeze(-1), dxdy,order).to(device)
        self.h = h
        self.lbdf1 = self.lbdf2 = self.lbdf3 = 1.0
        self.lbdubc = self.lbdvbc = 1.0
        self.B = torch.tensor(B).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.u_bc = torch.tensor(U_bc[:, 0:1]).double().to(device)
        self.v_bc = torch.tensor(U_bc[:, 1:2]).double().to(device)
        self.Re = torch.tensor(Re).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.dxdy = dxdy
        # 定义一个深度网络
        self.dnn = DNN(layers,layers_u,layers_v,layers_p).to(device)

        self.optimizer_LBFGS = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=LBFGS_iter,
            max_eval=LBFGS_iter,
            history_size=50,
            tolerance_grad=1.0 * np.finfo(float).eps,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), 0.001)
        self.Adam_iter = Adam_iter
        self.iter = 0
        self.loss = []

    def net_u(self, x, y):
        X = torch.cat([x, y], dim=1)
        X = 2.0 * (X-self.lb)/(self.ub - self.lb) - 1.0
        X = torch.cat([torch.cos(X @ self.B),torch.sin(X @ self.B)], dim=1)
        u = self.dnn(X)[:,0:3]
        return u

    def net_fSK(self, x, y):
        uvp = self.net_u(x,y)
        u_d = R_compute_field_gradients(uvp[:, 0:1], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        v_d = R_compute_field_gradients(uvp[:, 1:2], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        p_d = R_compute_field_gradients(uvp[:, 2:3], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        e1 = u_d[:, 0:1] + v_d[:, 1:2]
        e2 = uvp[:, 0:1] * u_d[:, 0:1] + uvp[:, 1:2] * u_d[:, 1:2] + p_d[:, 0:1] - 1 / self.Re * (u_d[:, 2:3] + u_d[:, 4:5])
        e3 = uvp[:, 0:1] * v_d[:, 0:1] + uvp[:, 1:2] * v_d[:, 1:2] + p_d[:, 1:2] - 1 / self.Re * (v_d[:, 2:3] + v_d[:, 4:5])
        return e1,e2,e3

    def net_fAD(self, x, y):
        u = self.net_u(x, y)[:,0:1]
        v = self.net_u(x, y)[:,1:2]
        p = self.net_u(x, y)[:,2:3]
        p_x = torch.autograd.grad(p, x,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0]
        p_y = torch.autograd.grad(p, y,grad_outputs=torch.ones_like(p),retain_graph=True,create_graph=True)[0]
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v, x,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_y = torch.autograd.grad(v, y,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y,grad_outputs=torch.ones_like(v_y),retain_graph=True,create_graph=True)[0]
        e1 = u_x + v_y
        e2 = u * u_x + v * u_y + p_x - 1 / self.Re * (u_xx + u_yy)
        e3 = u * v_x + v * v_y + p_y - 1 / self.Re * (v_xx + v_yy)
        return e1,e2,e3

    def Calculate_loss(self):
        bc = self.net_u(self.x_bc, self.y_bc)[:,0:2]
        f_e1AD, f_e2AD, f_e3AD = self.net_fAD(self.x_AD, self.y_AD)
        f_e1SK, f_e2SK, f_e3SK = self.net_fSK(self.x_SK, self.y_SK)
        loss_ubc = torch.mean((bc[:,0:1] - self.u_bc) ** 2)
        loss_vbc = torch.mean((bc[:,1:2] - self.v_bc) ** 2)
        loss_fe1 = torch.mean(torch.vstack((f_e1AD, f_e1SK)) ** 2)
        loss_fe2 = torch.mean(torch.vstack((f_e2AD, f_e2SK)) ** 2)
        loss_fe3 = torch.mean(torch.vstack((f_e3AD, f_e3SK)) ** 2)
        loss = self.lbdubc * loss_ubc + self.lbdvbc * loss_vbc + self.lbdf1 * loss_fe1 + self.lbdf2 * loss_fe2 + self.lbdf3 * loss_fe3
        if self.iter % 100 == 0 and self.iter < self.Adam_iter:
            g_ubc = self.Calculate_loss_gradient(loss_ubc)
            g_vbc = self.Calculate_loss_gradient(loss_vbc)
            g_fe1 = self.Calculate_loss_gradient(loss_fe1)
            g_fe2 = self.Calculate_loss_gradient(loss_fe2)
            g_fe3 = self.Calculate_loss_gradient(loss_fe3)
            sum_dlg = g_ubc + g_vbc + g_fe1 + g_fe2 + g_fe3
            self.lbdubc = 0.9 * self.lbdubc +0.1 * sum_dlg / g_ubc
            self.lbdvbc = 0.9 * self.lbdvbc +0.1 * sum_dlg / g_vbc
            self.lbdf1 = 0.9 * self.lbdf1 +0.1 * sum_dlg / g_fe1
            self.lbdf2 = 0.9 * self.lbdf2 +0.1 * sum_dlg / g_fe2
            self.lbdf3 =0.9 * self.lbdf3 +0.1 * sum_dlg / g_fe3
        self.iter += 1
        if self.iter % 10 == 0:
            print(
                'Iter %d, Loss: %.5e, loss_ubc: %.5e, loss_vbc: %.5e, loss_fe1: %.5e, Loss_fe2: %.5e, Loss_fe3: %.5e\n'
                'lbdubc: %.5e, lbdvbc: %.5e, lbdf1: %.5e, lbdf2: %.5e, lbdf3: %.5e, '
                % (self.iter, loss.item(), loss_ubc.item(), loss_vbc.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item(),
                   self.lbdubc, self.lbdvbc, self.lbdf1, self.lbdf2, self.lbdf3)
            )
        self.loss.append([self.iter, loss.item(), loss_ubc.item(), loss_vbc.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item()])
        return loss

    def Calculate_loss_gradient(self,loss_component):
        d_Lg = 0.0
        self.optimizer_adam.zero_grad()
        loss_component.backward(retain_graph=True)
        for name, param in self.dnn.named_parameters():
            if 'weight' in name and param.grad is not None:
                d_Lg += param.grad.norm(2).item() ** 2
        d_Lg = d_Lg ** 0.5
        return d_Lg

    def loss_func(self):
        self.optimizer_LBFGS.zero_grad()
        loss = self.Calculate_loss()
        loss.backward()
        return loss

    def train(self):
        self.dnn.train()
        # 先使用Adam预训练
        for i in range(1,self.Adam_iter):
            self.optimizer_adam.zero_grad()
            loss = self.Calculate_loss()
            loss.backward()
            self.optimizer_adam.step()
        # 再使用LBGFS
        self.optimizer_LBFGS.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).double().to(device)
        self.dnn.eval()
        u = self.net_u(x, y)[:,0:1]
        v = self.net_u(x, y)[:,1:2]
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        return u, v

# 定义边界点的数量
def bc_point(num_points, C0=50):
    # 生成在 [0, 1] x [0, 1] 方形域边界上等距分布的点
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    # 生成边界上的点
    x_boundary = np.concatenate((x[1:-1], x[1:-1], np.zeros(num_points)[1:-1], np.ones(num_points)[1:-1]))
    y_boundary = np.concatenate((np.zeros(num_points)[1:-1], np.ones(num_points)[1:-1], y[1:-1], y[1:-1]))
    # 修改上边界的 u 值
    u_boundary_top = 1 - np.cosh(C0 * (x[1:-1] - 0.5)) / np.cosh(0.5 * C0)
    u_boundary = np.concatenate(
        (np.zeros(num_points)[1:-1], u_boundary_top, np.zeros(num_points)[1:-1], np.zeros(num_points)[1:-1]))
    v_boundary = np.concatenate((np.zeros(num_points)[1:-1], np.zeros(num_points)[1:-1], np.zeros(num_points)[1:-1],
                                 np.zeros(num_points)[1:-1]))
    # 使用column_stack将x和y坐标组合成点
    return np.column_stack((x_boundary, y_boundary)), np.column_stack((u_boundary, v_boundary))


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
                       [0, 0, 0, 0, 0, 2 / dxdy]], dtype=torch.double)
    H0 = torch.nn.functional.pad(H, (0, terms_num-H.shape[1]), value=0).to(device)
    matrix = torch.matmul(moment_vector.unsqueeze(3), moment_vector.unsqueeze(2)) * kernel.unsqueeze(-1)
    matrix_sum = torch.sum(matrix, dim=1)
    matrix_inverse = torch.inverse(matrix_sum)
    C = torch.matmul(torch.matmul(moment_vector, matrix_inverse), H0.t().view(1, 1, terms_num, -1)).squeeze(0)
    return C

def R_compute_field_gradients(f_values,kernel,neighborhoods,C):
    u_ngr = f_values[neighborhoods][:,:,0]*(neighborhoods != -1)
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(u_ngr.unsqueeze(-1)*C*kernel,dim = 1)
    return field_gradients

## 主函数
dx = dy = 0.004
dxdy = dx*dy
d = 1.0
Re = 400
N1 = 2
N2 = 128
top_left_range = 0.1
top_right_range = 0.1
B = np.random.normal(0, 2, (N1,N2))
layers = [2*N2]+[20]*3
layers_u =[20]*3+[1]
layers_v =[20]*3+[1]
layers_p = [20]*3+[1]
ub = np.array([d,d])
lb = np.array([0,0])
x_bc,u_bc = bc_point(400)

x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
left_upper_points = x_area[(x_area[:, 0] < top_left_range) & (x_area[:, 1] > (ub[1] - top_left_range))]
right_upper_points = x_area[(x_area[:, 0] > (ub[0] - top_right_range)) & (x_area[:, 1] > (ub[1] - top_right_range))]
AD_points = np.vstack((left_upper_points, right_upper_points))
SK_points = x_area[~((x_area[:, 0] < top_left_range) & (x_area[:, 1] > (ub[1] - top_left_range)) |
                            (x_area[:, 0] > (ub[0] - top_right_range)) & (
                                    x_area[:, 1] > (ub[1] - top_right_range)))]
h = dx*1.4
Adam_iter = 2000
LBFGS_iter = 4000
order = 2
neighborhoods, distances, distance_vectors = find_neighborhood_points(SK_points, 2.0*h)
model = PhysicsInformedNN(AD_points,SK_points, neighborhoods, distances, distance_vectors,h,Re,B,order,
                 x_bc,u_bc, dxdy, layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')


mat = pd.read_excel('Re400.xlsx').to_numpy()
V_Ture = mat[:,2:3]
X_area = np.hstack((mat[:,0:1],mat[:,1:2]))
u_pred,v_pred = model.predict(X_area)
V_P = (u_pred**2+v_pred**2)**0.5
# np.save('V_PSKandAD.npy', V_P)
print("L2 error:", np.linalg.norm(V_Ture-V_P)/np.linalg.norm(V_Ture))

# plt.figure(figsize=(6, 6))
# plt.scatter(X_area[:,0], X_area[:,1], c = V_Ture[:,0],s=0.6, cmap='jet',vmax=1.0,vmin=0.0)
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.colorbar()
# plt.show()
# plt.figure(figsize=(6, 6))
# plt.scatter(X_area[:,0], X_area[:,1], c = V_P[:,0],s=0.6, cmap='jet',vmax=1.0,vmin=0.0)
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.colorbar()
# plt.show()
# plt.figure(figsize=(6, 6))
# plt.scatter(X_area[:,0], X_area[:,1], c = np.abs(V_Ture[:,0]-V_P[:,0]),s=0.6, cmap='jet',vmax=0.01,vmin=0.0)
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.yscale('log')
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,1],ls="-",lw=2,label="loss")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,2],ls="-",lw=2,label="loss_ubc")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,3],ls="-",lw=2,label="loss_vbc")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,4],ls="-",lw=2,label="loss_f1")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,5],ls="-",lw=2,label="loss_f2")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,6],ls="-",lw=2,label="loss_f3")
# plt.legend()
# plt.grid(linestyle=":")
# plt.axvline(x=1000,c="b",ls="--",lw=2)
# plt.xlim(0,np.array(model.loss)[-1,0])
# plt.show()