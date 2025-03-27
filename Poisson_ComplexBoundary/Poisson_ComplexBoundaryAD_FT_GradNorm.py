from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
import time
import shapely.geometry
import math

# 定义子网络列表
def NN_list(layers):
    depth = len(layers) - 1
    activation = torch.nn.Tanh
    layer_list = list()
    for i in range(depth - 1):
        linear_layer = torch.nn.Linear(layers[i], layers[i + 1])
        # Xavier 初始化
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
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.layers = torch.nn.Sequential(NN_list(layers)).double()

    def forward(self, x):
        out = self.layers(x)
        return out

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PhysicsInformedNN():
    # 初始化
    def __init__(self, x_area,x_bc,u_bc, layers,ub,lb,B,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.lbdbc = 1.0
        self.lbdf = 1.0
        self.B = torch.tensor(B).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.u_bc = torch.tensor(u_bc).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        # 定义一个深度网络
        self.dnn = DNN(layers).to(device)

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
        X = torch.cat([torch.cos(X @ self.B), torch.sin(X @ self.B)], dim=1)
        u = self.dnn(X)[:,0:1]
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        q = torch.exp(x)+torch.exp(y)
        f = u_xx + u_yy - q
        return f

    def Calculate_loss(self):
        bc = self.net_u(self.x_bc, self.y_bc)
        f_pde = self.net_f(self.x, self.y)
        loss_bc = torch.mean((bc-self.u_bc) ** 2)
        loss_f = torch.mean(f_pde ** 2)
        loss = self.lbdbc * loss_bc + self.lbdf * loss_f
        self.iter += 1
        if self.iter % 100 == 0 and self.iter < self.Adam_iter:
            sum_dlg = self.Calculate_loss_gradient(loss_bc) + self.Calculate_loss_gradient(loss_f)
            self.lbdf = 0.9 * self.lbdf +0.1 * sum_dlg / self.Calculate_loss_gradient(loss_f)
            self.lbdbc = 0.9 * self.lbdbc +0.1 * sum_dlg / self.Calculate_loss_gradient(loss_bc)



        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_f: %.5e, lbdbc: %.5e, lbdf: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_f.item(), self.lbdbc, self.lbdf)
            )
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_f.item()])
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
        loss.backward(retain_graph=True)
        return loss

    def train(self):
        self.dnn.train()
        # 先使用Adam预训练
        for i in range(1,self.Adam_iter):
            self.optimizer_adam.zero_grad()
            loss = self.Calculate_loss()
            loss.backward(retain_graph=True)
            self.optimizer_adam.step()
        # 再使用LBGFS
        self.optimizer_LBFGS.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).double().to(device)
        self.dnn.eval()
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        f_pde = self.net_f(x, y)
        u = u.detach().cpu().numpy()
        u_x = u_x.detach().cpu().numpy()
        u_y = u_y.detach().cpu().numpy()
        f_pde = f_pde.detach().cpu().numpy()
        return u, u_x, u_y, f_pde

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

def sph_kernel_gradient(distances, distance_vectors,dxdy, h):
    q = distances / h
    kernel_gradients = torch.zeros_like(distances, dtype=torch.float64).to(device)
    within_range = (0 <= q) & (q <= 2)
    kernel_gradients[within_range] = (15 / (7 * np.pi * h ** 3)) * (
            (-2.0 * q[within_range] + 1.5 * q[within_range] ** 2) * (q[within_range] <= 1) +
            (-0.5 * (2 - q[within_range]) ** 2) * ((1 < q[within_range]) & (q[within_range] <= 2))
    )
    kernel_GradVector = torch.einsum('ijk,ij->ijk', distance_vectors, -  kernel_gradients * dxdy / (distances + 1e-10))
    return kernel_GradVector

def compute_field_gradients(f_values,kernel_GradVector, neighborhoods):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',kernel_GradVector,u_ngr),dim = 1)
    return field_gradients

def CSPH_Correction_Factor(kernel_GradVector,distance_vectors):
    Correction_Factor = torch.einsum('ijk,ijk->ik',distance_vectors,kernel_GradVector)
    return Correction_Factor

def CSPH_compute_field_gradients(f_values,kernel_GradVector, neighborhoods,Correction_Factor):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',kernel_GradVector, (u_ngr-f_values)),dim = 1)
    CSPH_field_gradients = field_gradients/Correction_Factor
    return CSPH_field_gradients

def KGC_Matrix_inverse(kernel_GradVector,distance_vectors):
    Moment_Vector1 = torch.cat([distance_vectors[:, :, 0:1],distance_vectors[:, :, 1:2],], dim=2)
    Matrix = torch.matmul(kernel_GradVector.unsqueeze(3), Moment_Vector1.unsqueeze(2))
    Matrix_sum = torch.sum(Matrix, dim=1)
    Matrix_inverse = torch.inverse(Matrix_sum)
    return Matrix_inverse

def KGC_compute_field_gradients(f_values,kernel_GradVector, neighborhoods,Matrix_inverse):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',kernel_GradVector,(u_ngr-f_values)),dim = 1)
    FPM_field_gradients = torch.matmul(Matrix_inverse, field_gradients.unsqueeze(-1)).squeeze(-1)
    return FPM_field_gradients

# 有限核粒子方法
# 先需要根据distance_vectors（维度N,n,2）组一个向量列表（维度N,n,最后一个维度根据需求调整），然后做求逆操作
def FPM_Matrix_inverse(kernel,kernel_GradVector,distance_vectors,distances,dxdy):
    Moment_Vector1 = torch.cat([torch.ones(distances.shape).unsqueeze(-1).to(device),distance_vectors], dim=2)
    kernel_Vector = torch.cat([kernel.unsqueeze(-1) * dxdy,kernel_GradVector], dim=2)
    Matrix = torch.matmul(kernel_Vector.unsqueeze(3), Moment_Vector1.unsqueeze(2))
    Matrix_sum = torch.sum(Matrix, dim=1)
    Matrix_inverse = torch.inverse(Matrix_sum)
    return Matrix_inverse,kernel_Vector

def FPM_compute_field_gradients(f_values,kernel_Vector, neighborhoods,Matrix_inverse):
    u_ngr = f_values[neighborhoods][:,:,0]*(neighborhoods != -1)
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',kernel_Vector,(u_ngr-f_values)),dim = 1)
    FPM_field_gradients = torch.matmul(Matrix_inverse[:,1:3,:], field_gradients.unsqueeze(-1)).squeeze(-1)
    return FPM_field_gradients

# 再生核粒子方法
def Compute_C(distance_vectors,kernel,dxdy):
    Moment_Vector = torch.cat([torch.ones(kernel.shape).to(device),
                               distance_vectors[:, :, 0:1]/dxdy**0.5,
                               distance_vectors[:, :, 1:2]/dxdy**0.5,
                                distance_vectors[:, :, 0:1] * distance_vectors[:, :, 0:1]/dxdy,
                                distance_vectors[:, :, 0:1] * distance_vectors[:, :, 1:2]/dxdy,
                                distance_vectors[:, :, 1:2] * distance_vectors[:, :, 1:2]/dxdy], dim=2)
    H0 = torch.tensor([[0, 1 / dxdy ** 0.5, 0, 0, 0, 0],
                       [0, 0, 1 / dxdy ** 0.5, 0, 0, 0],
                       [0, 0, 0, 2 / dxdy, 0, 0],
                       [0, 0, 0, 0, 1 / dxdy, 0],
                       [0, 0, 0, 0, 0, 2 / dxdy]], dtype=torch.double).to(device)
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

# 将边界外的点删除,默认删除内部的点
def delete_point(x_area,bc,d_inside = True):
    polygon = shapely.geometry.Polygon(bc)
    points = shapely.geometry.MultiPoint(x_area)
    flag = np.ones(x_area.shape[0], dtype = np.bool)
    for i in range(x_area.shape[0]):
        flag[i] = polygon.covers(points[i])
    ff = np.where(flag == d_inside)
    x_area = np.delete(x_area, ff, axis=0)
    return x_area

# 定义要绘制的函数
def u_function(x, y):
    return np.exp(x)+np.exp(y)

## 主函数
dx = dy = 0.1
d = 4.0
N1 = 2
N2 = 20
B = np.random.normal(0, 1, (N1,N2))
layers = [2*N2]+[40]+[1]
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x = np.arange(lb[0]-dx/2,ub[0]+dx/2,dx)
y = np.arange(lb[1]-dy/2,ub[1]+dy/2,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
jiaodu = np.arange(0,2*math.pi,0.04)
r = 1.5 + 0.14*np.sin(4*jiaodu)+0.12*np.cos(6*jiaodu)+0.09*np.cos(5*jiaodu)
x_bce = 0.02+r*np.cos(jiaodu)
y_bce = r*np.sin(jiaodu)
x_bc = np.vstack((x_bce,y_bce)).T
u_bc = u_function(x_bc[:,0], x_bc[:,1]).reshape(-1, 1)
x_area = delete_point(x_area,x_bc,False)

Adam_iter = 1000
LBFGS_iter = 1000
model = PhysicsInformedNN(x_area,x_bc,u_bc, layers,ub,lb, B,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", end - start, 'seconds')
u_pred = model.predict(x_area)[0]
print("MSE:", np.mean(np.square(u_pred[:, 0] - u_function(x_area[:,0], x_area[:,1]))))
print("L2 error:", np.linalg.norm(u_function(x_area[:,0], x_area[:,1]) - u_pred[:, 0])/np.linalg.norm(u_function(x_area[:,0], x_area[:,1])))
# jiaodu = np.arange(0,2*math.pi,0.015)
# r = 1.5 + 0.14*np.sin(4*jiaodu)+0.12*np.cos(6*jiaodu)+0.09*np.cos(5*jiaodu)
# x_bce = 0.02+r*np.cos(jiaodu)
# y_bce = r*np.sin(jiaodu)
# x_bc = np.vstack((x_bce,y_bce)).T
# x_area = np.concatenate((x_area, x_bc), axis=0)
# u_pred = model.predict(x_area)[0]
# plt.figure()
# plt.scatter(x_area[:, 0], x_area[:, 1],c=np.abs(u_pred[:, 0]-u_function(x_area[:,0], x_area[:,1])),s=1.5, cmap='jet')
# plt.colorbar()
# plt.show()