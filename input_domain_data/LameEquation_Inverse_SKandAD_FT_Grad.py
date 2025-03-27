from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
import os
import scipy.io
import math
import matplotlib.tri as tri
import shapely.geometry
import pandas as pd
import time

# 定义子网络列表
def NN_list(layers):
    depth = len(layers) - 1
    activation = torch.nn.Tanh
    layer_list = list()
    for i in range(depth - 1):
        layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
        layer_list.append(('activation_%d' % i, activation()))
    layer_list.append(
        ('layer_%d' % (depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
    layerDict = OrderedDict(layer_list)
    return layerDict

# 定义网络
class DNN(torch.nn.Module):
    def __init__(self,layers,layers_S,layers_Lame):
        super(DNN, self).__init__()
        self.layers = torch.nn.Sequential(NN_list(layers)).double()
        self.layers_S = torch.nn.Sequential(NN_list(layers_S)).double()
        self.layers_Lame = torch.nn.Sequential(NN_list(layers_Lame)).double()

    def forward(self, x):
        out = self.layers(x)
        out = torch.cat([self.layers_S(out),self.layers_Lame(out)], dim=1)
        return out

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PhysicsInformedNN():
    # 初始化
    def __init__(self, x_area, neighborhoods, distances, distance_vectors,h,B,
                 X_leftright, Sxx_leftright, X_topbottom, Syy_topbottom, E_Training,
                 dxdy, layers, layers_S, layers_Lame, ub, lb, Adam_iter, LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.neighborhoods = neighborhoods.to(device)
        self.distances = distances.to(device)
        self.distance_vectors = distance_vectors.to(device)
        self.kernel = sph_kernel(distances, h).to(device)
        self.RKPM_C = Compute_C(self.distance_vectors, self.kernel.unsqueeze(-1), dxdy).to(device)
        self.h = h
        self.lbdleftright = self.lbdtopbottom = self.lbdfe1 = self.lbdfe2 = self.lbdfe3 = self.lbdfe4 = self.lbdfe5  = 1.0
        self.B = torch.tensor(B).double().to(device)
        self.X_leftright = torch.tensor(X_leftright, requires_grad=True).double().to(device)
        self.X_topbottom = torch.tensor(X_topbottom, requires_grad=True).double().to(device)
        self.Sxx_leftright = torch.tensor(Sxx_leftright).double().to(device)
        self.Syy_topbottom = torch.tensor(Syy_topbottom).double().to(device)
        self.E_Training = torch.tensor(E_Training).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.dxdy = dxdy
        # 定义一个深度网络
        self.dnn = DNN(layers, layers_S, layers_Lame).to(device)

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
        u = self.dnn(X)[:,0:5]
        return u

    def net_f(self, x, y):
        S_andLame = self.net_u(x, y)
        e1 = (2 * S_andLame[:, 4:5] + S_andLame[:, 3:4]) * self.E_Training[:, 0:1] \
             + S_andLame[:, 3:4] * self.E_Training[:, 2:3] - S_andLame[:,0:1]
        e2 = (2 * S_andLame[:, 4:5] + S_andLame[:, 3:4]) * self.E_Training[:, 2:3] \
             + S_andLame[:, 3:4] * self.E_Training[:, 0:1] - S_andLame[:,2:3]
        e3 = 2 * S_andLame[:, 4:5] * self.E_Training[:, 1:2] - S_andLame[:,1:2]
        # AD
        # Sxx_x = torch.autograd.grad(S_andLame[:, 0:1], x, grad_outputs=torch.ones_like(S_andLame[:, 0:1]),retain_graph=True,create_graph=True)[0]
        # Sxy_x = torch.autograd.grad(S_andLame[:, 1:2], x, grad_outputs=torch.ones_like(S_andLame[:, 1:2]),retain_graph=True,create_graph=True)[0]
        # Sxy_y = torch.autograd.grad(S_andLame[:, 1:2], y, grad_outputs=torch.ones_like(S_andLame[:, 1:2]),retain_graph=True,create_graph=True)[0]
        # Syy_y = torch.autograd.grad(S_andLame[:, 2:3], y, grad_outputs=torch.ones_like(S_andLame[:, 2:3]),retain_graph=True, create_graph=True)[0]
        # e4 = Sxx_x + Sxy_y
        # e5 = Sxy_x + Syy_y

        ## RKPM
        Sxx = R_compute_field_gradients(S_andLame[:, 0:1], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        Sxy = R_compute_field_gradients(S_andLame[:, 1:2], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        Syy = R_compute_field_gradients(S_andLame[:, 2:3], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        e4 = Sxx[:,0:1] + Sxy[:,1:2]
        e5 = Sxy[:,0:1] + Syy[:,1:2]

        return e1,e2,e3,e4,e5

    def Calculate_loss(self):
        bc_leftright = self.net_u(self.X_leftright[:, 0:1], self.X_leftright[:, 1:2])[:,0:1]
        bc_topbottom = self.net_u(self.X_topbottom[:, 0:1], self.X_topbottom[:, 1:2])[:,2:3]
        f_e1,f_e2,f_e3,f_e4,f_e5 = self.net_f(self.x, self.y)
        loss_bc_leftright = torch.mean((bc_leftright - self.Sxx_leftright) ** 2)
        loss_bc_topbottom = torch.mean((bc_topbottom - self.Syy_topbottom) ** 2)
        loss_fe1 = torch.mean(f_e1 ** 2)
        loss_fe2 = torch.mean(f_e2 ** 2)
        loss_fe3 = torch.mean(f_e3 ** 2)
        loss_fe4 = torch.mean(f_e4 ** 2)
        loss_fe5 = torch.mean(f_e5 ** 2)
        loss = self.lbdleftright * loss_bc_leftright + self.lbdtopbottom * loss_bc_topbottom + \
               self.lbdfe1 * loss_fe1 + self.lbdfe2 * loss_fe2 + self.lbdfe3 * loss_fe3 + \
               self.lbdfe4 * loss_fe4 + self.lbdfe5 * loss_fe5
        self.iter += 1
        if self.iter % 1000 == 0 and self.iter < self.Adam_iter:
            g_bc_leftright = self.Calculate_loss_gradient(loss_bc_leftright)
            g_bc_topbottom = self.Calculate_loss_gradient(loss_bc_topbottom)
            g_fe1 = self.Calculate_loss_gradient(loss_fe1)
            g_fe2 = self.Calculate_loss_gradient(loss_fe2)
            g_fe3 = self.Calculate_loss_gradient(loss_fe3)
            g_fe4 = self.Calculate_loss_gradient(loss_fe4)
            g_fe5 = self.Calculate_loss_gradient(loss_fe5)
            sum_dlg = g_bc_leftright + g_bc_topbottom + g_fe1 + g_fe2 + g_fe3 + g_fe4 + g_fe5
            self.lbdleftright = 0.9 * self.lbdleftright + 0.1 * sum_dlg / g_bc_leftright
            self.lbdtopbottom = 0.9 * self.lbdtopbottom + 0.1 * sum_dlg / g_bc_topbottom
            self.lbdfe1 = 0.9 * self.lbdfe1 + 0.1 * sum_dlg / g_fe1
            self.lbdfe2 = 0.9 * self.lbdfe2 +0.1 * sum_dlg / g_fe2
            self.lbdfe3 = 0.9 * self.lbdfe3 +0.1 * sum_dlg / g_fe3
            self.lbdfe4 = 0.9 * self.lbdfe4 +0.1 * sum_dlg / g_fe4
            self.lbdfe5 = 0.9 * self.lbdfe5 +0.1 * sum_dlg / g_fe5
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, loss_bc_leftright: %.5e, loss_bc_topbottom: %.5e, loss_fe1: %.5e, Loss_fe2: %.5e, Loss_fe3: %.5e, Loss_fe4: %.5e, Loss_fe5: %.5e\n'
                'lbdleftright: %.5e, lbdtopbottom: %.5e,lbdfe1: %.5e,lbdfe2: %.5e, lbdfe3: %.5e, lbdfe4: %.5e, lbdfe5: %.5e, '
                % (self.iter, loss.item(), loss_bc_leftright.item(), loss_bc_topbottom.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item(), loss_fe4.item(), loss_fe5.item(),
                   self.lbdleftright, self.lbdtopbottom,self.lbdfe1,self.lbdfe2, self.lbdfe3, self.lbdfe4, self.lbdfe5)
            )
        self.loss.append([self.iter, loss.item(), loss_bc_leftright.item(), loss_bc_topbottom.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item(), loss_fe4.item(), loss_fe5.item()])
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
        Sxx = self.net_u(x, y)[:, 0:1]
        Sxy = self.net_u(x, y)[:, 1:2]
        Syy = self.net_u(x, y)[:, 2:3]
        Lbd = self.net_u(x, y)[:, 3:4]
        M = self.net_u(x, y)[:, 4:5]
        Sxx = Sxx.detach().cpu().numpy()
        Sxy = Sxy.detach().cpu().numpy()
        Syy = Syy.detach().cpu().numpy()
        Lbd = Lbd.detach().cpu().numpy()
        M = M.detach().cpu().numpy()
        return Sxx, Sxy,Syy,Lbd,M

# 领域点搜索
def find_neighborhood_points(coords, k_neighbors=None, radius=None):
    # 创建一个 cKDTree
    kdtree = cKDTree(coords)
    # 查询每个点的邻域点
    neighborhoods = []
    distances = []
    distance_vectors = []
    neighborhood_radius = []
    for i, point in enumerate(coords):
        if k_neighbors:
            # 查询指定数量的最近邻点
            neighbors_indices = kdtree.query(point, k_neighbors)[1]
        elif radius:
            # 查询在指定半径内的邻域点
            neighbors_indices = kdtree.query_ball_point(point, radius)
        else:
            raise ValueError("You must specify either 'k_neighbors' or 'radius'.")
        # 获取邻域点的坐标
        neighbors_coords = coords[neighbors_indices]
        # 计算邻域点与当前点的距离
        dist = np.linalg.norm(neighbors_coords - point, axis=1)
        # 计算距离向量
        dist_vector = neighbors_coords - point
        # 计算邻域半径
        if len(neighbors_indices) > 0:
            neighbor_radius = np.max(np.linalg.norm(neighbors_coords - point, axis=1))
        else:
            neighbor_radius = 0.0
        neighborhoods.append(torch.tensor(neighbors_indices, dtype=torch.long))
        distances.append(torch.tensor(dist, dtype=torch.double))
        distance_vectors.append(torch.tensor(dist_vector, dtype=torch.double))
        neighborhood_radius.append(torch.tensor(neighbor_radius, dtype=torch.double))

    # 对结果进行填充
    neighborhoods = pad_sequence(neighborhoods, batch_first=True, padding_value=-1)
    distances = pad_sequence(distances, batch_first=True, padding_value=-1)
    distance_vectors = pad_sequence(distance_vectors, batch_first=True, padding_value=0)
    neighborhood_radius = torch.stack(neighborhood_radius)

    return neighborhoods, distances, distance_vectors, neighborhood_radius


def sph_kernel(distances, h):
    q = distances / h[:, None]
    result = torch.zeros_like(distances, dtype=torch.float64)
    # 计算核函数的值
    within_range = (0 <= q) & (q <= 2)
    result[within_range] = (
        (2/3 - 1.0 * q[within_range] ** 2 + 0.5 * q[within_range] ** 3) * (q[within_range] <= 1) +
        (1/6 * (2 - q[within_range]) ** 3) * ((1 < q[within_range]) & (q[within_range] <= 2))
    )
    result = (15 / (7 * np.pi * h[:, None] ** 2)) * result
    return result

# 再生核粒子方法
def Compute_C(distance_vectors,kernel,dxdy):
    Moment_Vector = torch.cat([torch.ones(kernel.shape).to(device),
                               distance_vectors[:, :, 0:1]/dxdy**0.5,
                               distance_vectors[:, :, 1:2]/dxdy**0.5,
                                distance_vectors[:, :, 0:1] * distance_vectors[:, :, 0:1]/dxdy,
                                distance_vectors[:, :, 0:1] * distance_vectors[:, :, 1:2]/dxdy,
                                distance_vectors[:, :, 1:2] * distance_vectors[:, :, 1:2]/dxdy], dim=2)
    H0 = torch.tensor([[0, 1 / dxdy ** 0.5, 0, 0, 0, 0],
                       [0, 0, 1 / dxdy ** 0.5, 0, 0, 0]], dtype=torch.double).to(device)
    Matrix = torch.matmul(Moment_Vector.unsqueeze(3), Moment_Vector.unsqueeze(2))*kernel.unsqueeze(-1)
    Matrix_sum = torch.sum(Matrix, dim=1)
    Matrix_inverse = torch.inverse(Matrix_sum)
    C = torch.matmul(torch.matmul(Moment_Vector, Matrix_inverse), H0.t().view(1, 1, 6, -1)).squeeze(0)
    return C

def R_compute_field_gradients(f_values,kernel,neighborhoods,C):
    u_ngr = f_values[neighborhoods][:,:,0]*(neighborhoods != -1)
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(u_ngr.unsqueeze(-1)*C*kernel,dim = 1)
    return field_gradients

# 主函数
# folder_path = 'tumor_example_data/'
folder_path = 'soft_background_example_data/'
# folder_path = 'stiff_background_example_data/'

Exx_Training = np.genfromtxt(os.path.join(folder_path, 'Exx_structured.txt'), delimiter=',').T
Eyy_Training = np.genfromtxt(os.path.join(folder_path, 'Eyy_structured.txt'), delimiter=',').T
Exy_Training = np.genfromtxt(os.path.join(folder_path, 'Exy_structured.txt'), delimiter=',').T
Sxx_Training = np.genfromtxt(os.path.join(folder_path, 'Sxx_structured.txt'), delimiter=',').T
Syy_Training = np.genfromtxt(os.path.join(folder_path, 'Syy_structured.txt'), delimiter=',').T
Sxy_Training = np.genfromtxt(os.path.join(folder_path, 'Sxy_structured.txt'), delimiter=',').T
Youngs_GroundTruth = np.genfromtxt(os.path.join(folder_path, 'Youngs_structured.txt'), delimiter=',').T
Poissons_GroundTruth = np.genfromtxt(os.path.join(folder_path, 'Poissons_structured.txt'), delimiter=',').T

x = np.linspace(0, 0.076, Exx_Training.shape[0])
y = np.linspace(0, 0.108, Exx_Training.shape[1])
X_Training,Y_Training = np.meshgrid(x,y)
X_Training = X_Training.T
Y_Training = Y_Training.T
ub = np.array([np.max(X_Training),np.max(Y_Training)])
lb = np.array([np.min(X_Training),np.min(Y_Training)])


# Compute characteristic scales
Syy_Max = np.max(Syy_Training[-1:,:])
sigma0 = Syy_Max
l0 = np.mean(ub-lb)

dx = X_Training[1,0]-X_Training[0,0]
dy = Y_Training[0,1]-Y_Training[0,0]
dxdy = dx*dy
ub = ub/l0
lb = lb/l0
a = np.concatenate((X_Training[:,0:1],Y_Training[:,0:1]),axis = 1)
x_leftright = np.concatenate((np.concatenate((X_Training[0:1,:],Y_Training[0:1,:]),axis = 0),
                             np.concatenate((X_Training[-1:,:],Y_Training[-1:,:]),axis = 0)), axis = 1).T/l0
Sxx_leftright = np.zeros((x_leftright.shape[0], 1))/sigma0
x_topbottom = np.concatenate((np.concatenate((X_Training[:,0:1],Y_Training[:,0:1]),axis = 1),
                             np.concatenate((X_Training[:,-1:],Y_Training[:,-1:]),axis = 1)), axis = 0)/l0
Syy_topbottom = np.concatenate((Syy_Training[:,0:1],Syy_Training[:,-1:]), axis = 0)/sigma0
x_area = np.hstack((X_Training.reshape(-1,1),Y_Training.reshape(-1,1)))/l0
E_Training = np.concatenate(([Exx_Training.reshape(-1, 1),Exy_Training.reshape(-1, 1),Eyy_Training.reshape(-1, 1)]),axis = 1)
Adam_iter = 5000
LBFGS_iter = 5000
N1 = 2
N2 = 128
B = np.random.normal(0, 10, (N1,N2))
layers = [2*N2]+[20]*3
layers_S = [20]*3+[3]
layers_Lame = [20]*3+[2]
neighborhoods, distances, distance_vectors, neighborhood_radius = find_neighborhood_points(x_area,k_neighbors=13)
h = neighborhood_radius*1.1/2.0
model = PhysicsInformedNN(x_area, neighborhoods, distances, distance_vectors,h,B,
                          x_leftright,Sxx_leftright,x_topbottom,Syy_topbottom,E_Training,
                          dxdy,layers,layers_S,layers_Lame,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')

Sxx, Sxy, Syy,Lbd,M = model.predict(x_area)



# plt.figure(figsize=(3, 3))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=Exx_Training,s=5, cmap='jet')
# plt.colorbar()
# plt.figure(figsize=(3, 3))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=Exy_Training,s=5, cmap='jet')
# plt.colorbar()
# plt.figure(figsize=(3, 3))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=Eyy_Training,s=5, cmap='jet')
# plt.colorbar()
# plt.show()
# 画图
plt.figure(figsize=(3, 3))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Sxx,s=5, cmap='jet')
plt.colorbar()
plt.figure(figsize=(3, 3))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Sxy,s=5, cmap='jet',vmax=0.3,vmin=-0.3)
plt.colorbar()
plt.figure(figsize=(3, 3))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Sxy_Training.reshape(-1, 1)/sigma0,s=5, cmap='jet',vmax=0.3,vmin=-0.3)
plt.colorbar()
plt.figure(figsize=(3, 3))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Syy,s=5, cmap='jet',vmax=1.4,vmin=0.6)
plt.colorbar()
plt.figure(figsize=(3, 3))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Syy_Training.reshape(-1, 1)/sigma0,s=5, cmap='jet',vmax=1.4,vmin=0.6)
plt.colorbar()
# plt.figure(figsize=(3, 3))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=Eyy_Training.reshape(-1, 1),s=5, cmap='jet')
# plt.colorbar()

E = sigma0*M*(3*Lbd+2*M)/(Lbd+M)
v = Lbd/(2*(Lbd+M))
Youngs_GroundTruth.reshape(-1,1)/1000-E/(1-v**2)/1000
print("L2 error:", np.linalg.norm(Youngs_GroundTruth.reshape(-1,1)/1000-E/(1-v**2)/1000)/np.linalg.norm(Youngs_GroundTruth.reshape(-1,1)/1000))
print("L2 error:", np.linalg.norm(Poissons_GroundTruth.reshape(-1,1)-v/(1-v))/np.linalg.norm(Poissons_GroundTruth.reshape(-1,1)))
#
plt.figure(figsize=(5, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Youngs_GroundTruth.reshape(-1,1)/1000,s=5, cmap='jet', vmin=0.8, vmax=2.2)
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.figure(figsize=(5, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=E/(1-v**2)/1000,s=5, cmap='jet', vmin=0.8, vmax=2.2)
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.figure(figsize=(5, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=np.abs(Youngs_GroundTruth.reshape(-1,1)/1000-E/(1-v**2)/1000),s=5, cmap='jet')
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.figure(figsize=(5, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=Poissons_GroundTruth.reshape(-1,1),s=5, cmap='jet', vmin=0.3, vmax=0.5)
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.figure(figsize=(5, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=v/(1-v),s=5, cmap='jet', vmin=0.3, vmax=0.5)
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.figure(figsize=(5, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=np.abs(Poissons_GroundTruth.reshape(-1,1)-v/(1-v)),s=5, cmap='jet')
plt.axis('equal')
plt.axis('off')
plt.colorbar()
plt.show()

# # loss曲线
# plt.figure()
# plt.yscale('log')
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,1],ls="-",lw=2,label="loss")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,2],ls="-",lw=2,label="loss_bc_leftright")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,3],ls="-",lw=2,label="loss_bc_topbottom")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,4],ls="-",lw=2,label="loss_fe1")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,5],ls="-",lw=2,label="loss_fe2")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,6],ls="-",lw=2,label="loss_fe3")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,7],ls="-",lw=2,label="loss_fe4")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,8],ls="-",lw=2,label="loss_fe5")
# plt.legend()
# plt.grid(linestyle=":")
# plt.axvline(x=1000,c="b",ls="--",lw=2)
# plt.xlim(0,np.array(model.loss)[-1,0])
# plt.show()