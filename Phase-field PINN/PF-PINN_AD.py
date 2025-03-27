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
    def __init__(self, xt_area,x_ic,u_ic,x_bc1,x_bc2, layers,layers_u,layers_v,layers_p,layers_C,Adam_iter):
        self.xt_area = torch.tensor(xt_area, requires_grad=True).float().to(device)
        self.x_bc1 = torch.tensor(x_bc1, requires_grad=True).float().to(device)
        self.x_bc2 = torch.tensor(x_bc2, requires_grad=True).float().to(device)
        self.x_ic = torch.tensor(x_ic, requires_grad=True).float().to(device)
        self.u_ic = torch.tensor(u_ic).float().to(device)
        # 定义一个深度网络
        self.dnn = DNN(layers,layers_u,layers_v,layers_p,layers_C).to(device)

        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), 0.001)
        self.Adam_iter = Adam_iter
        self.iter = 0
        self.loss = []

    def net_u(self, xyt):
        u = self.dnn(xyt)
        return u

    def net_f(self, xyt):
        uvpc = self.net_u(xyt)
        C = uvpc[:, 3:4]*(torch.abs(uvpc[:, 3:4]) <= 1) + torch.sign(uvpc[:, 3:4])*(torch.abs(uvpc[:, 3:4]) > 1)
        rho = 0.5 * (1 + C) * 1000.0 + 0.5 * (1 - C) * 1.0
        mu = 0.5 * (1 + C) * 10.0 + 0.5 * (1 - C) * 0.1

        u_d = torch.autograd.grad(uvpc[:, 0:1], xyt, grad_outputs=torch.ones_like(uvpc[:, 0:1]), retain_graph=True, create_graph=True)[0]
        v_d = torch.autograd.grad(uvpc[:, 1:2], xyt, grad_outputs=torch.ones_like(uvpc[:, 1:2]), retain_graph=True, create_graph=True)[0]
        p_d = torch.autograd.grad(uvpc[:, 3:4], xyt, grad_outputs=torch.ones_like(uvpc[:, 1:2]), retain_graph=True,create_graph=True)[0][:, 0:2]
        c_d = torch.autograd.grad(uvpc[:, 3:4], xyt, grad_outputs=torch.ones_like(uvpc[:, 3:4]), retain_graph=True, create_graph=True)[0]
        u_dxx = torch.autograd.grad(u_d[:, 0:1], xyt, grad_outputs=torch.ones_like(u_d[:, 0:1]), retain_graph=True,create_graph=True)[0][:, 0:1]
        u_dyy = torch.autograd.grad(u_d[:, 1:2], xyt, grad_outputs=torch.ones_like(u_d[:, 1:2]), retain_graph=True,create_graph=True)[0][:, 1:2]
        v_dxx = torch.autograd.grad(v_d[:, 0:1], xyt, grad_outputs=torch.ones_like(v_d[:, 0:1]), retain_graph=True,create_graph=True)[0][:, 0:1]
        v_dyy = torch.autograd.grad(v_d[:, 1:2], xyt, grad_outputs=torch.ones_like(v_d[:, 1:2]), retain_graph=True,create_graph=True)[0][:, 1:2]
        c_dxx = torch.autograd.grad(c_d[:, 0:1], xyt, grad_outputs=torch.ones_like(c_d[:, 0:1]), retain_graph=True,create_graph=True)[0][:, 0:1]
        c_dyy = torch.autograd.grad(c_d[:, 1:2], xyt, grad_outputs=torch.ones_like(c_d[:, 1:2]), retain_graph=True,create_graph=True)[0][:, 1:2]
        fai = C*(C**2-1)-0.01**2 * (c_dxx+c_dyy)
        fsigma = (3 * 2 ** .5 / 4) * (1.96 / 0.01) * fai * c_d[:, 0:2]
        fai_d = torch.autograd.grad(fai, xyt, grad_outputs=torch.ones_like(fai), retain_graph=True, create_graph=True)[0][:, 0:2]
        fai_dxx = torch.autograd.grad(fai_d[:, 0:1], xyt, grad_outputs=torch.ones_like(fai_d[:, 0:1]), retain_graph=True, create_graph=True)[0][:, 0:1]
        fai_dyy = torch.autograd.grad(fai_d[:, 1:2], xyt, grad_outputs=torch.ones_like(fai_d[:, 1:2]), retain_graph=True, create_graph=True)[0][:, 1:2]

        e1 = u_d[:, 0:1] + v_d[:, 1:2]
        e2 = (rho * (u_d[:,2:3]+uvpc[:, 0:1] * u_d[:, 0:1] + uvpc[:, 1:2] * u_d[:, 1:2]) \
             + 1 *(p_d[:, 0:1]-0.5 * 9.9 * c_d[:,1:2]*(u_d[:,1:2]+v_d[:,0:1])-9.9*c_d[:,0:1]*u_d[:,0:1]-mu*(u_dxx + u_dyy)-fsigma[:,0:1]) - 0.0)/1000.0
        e3 = (rho * (v_d[:,2:3]+uvpc[:, 0:1] * v_d[:, 0:1] + uvpc[:, 1:2] * v_d[:, 1:2]) \
             + 1 *(p_d[:, 1:2]-0.5 * 9.9 * c_d[:,0:1]*(u_d[:,1:2]+v_d[:,0:1])-9.9*c_d[:,1:2]*v_d[:,1:2]-mu*(v_dxx + v_dyy)-fsigma[:,1:2]) - rho * 0.98)/1000.0
        e4 = c_d[:,2:3] + uvpc[:, 0:1]*c_d[:,0:1] + uvpc[:, 1:2]*c_d[:,1:2] - 1e-4 * (fai_dxx + fai_dyy)
        return e1,e2,e3,e4

    def net_bc2(self, xyt):
        uvpc = self.net_u(xyt)
        v_d = torch.autograd.grad(uvpc[:, 1:2], xyt, grad_outputs=torch.ones_like(uvpc[:, 1:2]), retain_graph=True, create_graph=True)[0]
        return uvpc[:, 0:1], v_d[:, 0:1]

    def Calculate_loss(self,x_ic_batch, u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch):
        ic = self.net_u(x_ic_batch)
        bc1 = self.net_u(x_bc1_batch)
        bc2u,bc2dvdx = self.net_bc2(x_bc2_batch)
        f_e1,f_e2,f_e3,f_e4 = self.net_f(xt_area_batch)
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

    def get_batch(self,T_N, batch_size_ic=1000, batch_size_bc1=300,
                  batch_size_bc2=300, batch_size_area=10000):
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
        total_points = n_i * (T_N*(T_N <= n_j)+n_j*(T_N > n_j))   # 总点数 80601 * 101
        k_batch = np.random.choice(total_points, size=batch_size_area, replace=False)
        j_batch = k_batch // n_i  # 计算 i 索引
        i_batch = k_batch % n_i  # 计算 j 索引
        xt_area_batch = self.xt_area[i_batch, j_batch, :]
        return x_ic_batch,u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch

    def train(self):
        self.dnn.train()
        # 使用Adam训练
        for i in range(1,self.Adam_iter):
            T_N = self.iter % 50 + 1
            x_ic_batch, u_ic_batch, x_bc1_batch, x_bc2_batch, xt_area_batch, i_batch, j_batch = self.get_batch(T_N)
            self.optimizer_adam.zero_grad()
            # xt_area_batch.requires_grad_(True)
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

layers = [3]+[100]*5
layers_u = [100]*5+[1]
layers_v = [100]*5+[1]
layers_p = [100]*5+[1]
layers_C = [100]*5+[1]

Adam_iter = 5000
model = PhysicsInformedNN(xt_area,x_ic,u_ic,x_bc1,x_bc2, layers,layers_u,layers_v,layers_p,layers_C,Adam_iter)

# model.dnn.load_state_dict(torch.load("modelDNNAD0303.pth"))
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')

torch.save(model.dnn.state_dict(), "modelDNNAD0303.pth")
print("网络参数已保存！")

x = np.arange(-0.5+0.0*dx,0.5+1.0*dx,4*dx)
y = np.arange(-1.0+0.0*dy,1.0+1.0*dy,4*dy)
t = np.arange(0.0*dt, 0.5+1.0*dt, 5*dt)
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
uic_pred,vic_pred,cic_pred = model.predict(x_ic)
plt.figure(figsize=(8, 6))
plt.scatter(x_ic[:,0], x_ic[:,1], c = cic_pred, cmap='jet')
plt.colorbar(label='Density')  # 添加颜色条
plt.title('2D Scatter Cloud Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# xt_area = xt_area.reshape(-1, len(t), 3)
# plt.figure(figsize=(8, 6))
# plt.scatter(xt_area[:,0,0], xt_area[:,0,1], c = u_pred.reshape(-1, len(t), 3)[:,-1,:], cmap='jet')
# plt.colorbar(label='Density')  # 添加颜色条
# plt.title('2D Scatter Cloud Plot')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.scatter(xt_area[:,0,0], xt_area[:,0,1], c = v_pred.reshape(-1, len(t), 3)[:,-1,:], cmap='jet')
# plt.colorbar(label='Density')  # 添加颜色条
# plt.title('2D Scatter Cloud Plot')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()
#
# plt.figure(figsize=(8, 6))
# plt.scatter(xt_area[:,0,0], xt_area[:,0,1], c = c_pred.reshape(-1, len(t), 3)[:,-1,:], cmap='jet')
# plt.colorbar(label='Density')  # 添加颜色条
# plt.title('2D Scatter Cloud Plot')
# plt.xlabel('X axis')
# plt.ylabel('Y axis')
# plt.show()