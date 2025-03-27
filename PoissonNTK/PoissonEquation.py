from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
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
    def __init__(self, x_area, neighborhoods, distances, distance_vectors,h,B,order,
                 x_bc,u_bc, dxdy, layers,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.neighborhoods = neighborhoods.to(device)
        self.distances = distances.to(device)
        self.distance_vectors = distance_vectors.to(device)
        self.kernel = sph_kernel(distances, h).to(device)
        self.RKPM_C = Compute_C(self.distance_vectors, self.kernel.unsqueeze(-1), dxdy,order).to(device)
        self.h = h
        self.lbdbc = self.lbdf = 1.0
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.u_bc = torch.tensor(u_bc).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.dxdy = dxdy
        self.B = torch.tensor(B).double().to(device)
        self.eigenvalues_iter = []
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
        self.L2 = []

    def net_u(self, x, y):
        X = torch.cat([x, y], dim=1)
        X = 2.0 * (X-self.lb)/(self.ub - self.lb) - 1.0
        # X = torch.cat([torch.cos(X @ self.B), torch.sin(X @ self.B)], dim=1)
        u = self.dnn(X)[:,0:1]
        return u

    def net_fAD(self, x, y):
        u = self.net_u(x,y)
        q = torch.sin(2 * np.pi * y) * (20 * torch.tanh(10 * x) * (10 * torch.tanh(10 * x) ** 2 - 10) - (
                    2 * (np.pi) ** 2 * torch.sin(2 * np.pi * x)) * 0.2) - 4 * (np.pi) ** 2 * torch.sin(
            2 * np.pi * y) * (torch.tanh(10 * x) + torch.sin(2 * np.pi * x) * 0.1)

        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]
        f = u_xx + u_yy - q
        return f

    def net_fSK(self, x, y):
        u = self.net_u(x, y)
        q = torch.sin(2 * np.pi * y) * (20 * torch.tanh(10 * x) * (10 * torch.tanh(10 * x) ** 2 - 10) - (
                2 * (np.pi) ** 2 * torch.sin(2 * np.pi * x)) * 0.2) - 4 * (np.pi) ** 2 * torch.sin(
            2 * np.pi * y) * (torch.tanh(10 * x) + torch.sin(2 * np.pi * x) * 0.1)

        u_d = R_compute_field_gradients(u, self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        f = u_d[:, 2:3] + u_d[:, 4:5] - q
        return f

    def NTK(self,f):
        params = list(self.dnn.parameters())  # 获取网络参数
        Jacobian_matrix = torch.zeros((f.numel(), sum(p.numel() for p in params)))  # 初始化 Jacobian 矩阵

        # 计算 Jacobian 矩阵
        for i in range(f.numel()):
            grads = torch.autograd.grad(f[i], params, retain_graph=True)
            grad_params = torch.cat([grad.flatten() for grad in grads])  # 将参数梯度拼接成一个向量
            Jacobian_matrix[i] = grad_params
        K = Jacobian_matrix @ Jacobian_matrix.T
        eigenvalues, _ = torch.linalg.eig(K)
        eigenvalues = eigenvalues.real.detach().cpu().numpy()
        return eigenvalues

    def Calculate_loss(self):
        f_bc = self.net_u(self.x_bc, self.y_bc) - self.u_bc
        f_pdeSK = self.net_fSK(self.x, self.y)
        # f_pdeAD = self.net_fAD(self.x, self.y)
        # fSK = torch.cat((f_pdeSK, f_bc), dim=0)
        # fAD = torch.cat((f_pdeAD, f_bc), dim=0)
        loss_bc = torch.mean(f_bc ** 2)
        loss_f = torch.mean(f_pdeSK ** 2)
        loss = self.lbdbc * loss_bc + self.lbdf * loss_f

        # if self.iter % 100 == 0 and self.iter < self.Adam_iter:
        #     g_bc = self.Calculate_loss_gradient(loss_bc)
        #     g_f = self.Calculate_loss_gradient(loss_f)
        #     sum_dlg = g_bc + g_f
        #     self.lbdbc = 0.9 * self.lbdbc +0.1 * sum_dlg / g_bc
        #     self.lbdf = 0.9 * self.lbdf +0.1 * sum_dlg / g_f

        # if self.iter % 10 == 0:
        #     l2 = self.predict(x_area)[1]
        #     self.L2.append([time.perf_counter()-start, l2])

        if self.iter % 100 == 0:
            # self.eigenvalues_iter.append(sorted(self.NTK(fAD), reverse=True))
            # self.eigenvalues_iter.append(sorted(self.NTK(fSK), reverse=True))
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_f: %.5e, lbdbc: %.5e, lbdf: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_f.item(), self.lbdbc, self.lbdf)
            )
        self.iter += 1
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_f.item()])
        return loss

    def Calculate_lossAD(self):
        f_bc = self.net_u(self.x_bc, self.y_bc) - self.u_bc
        f_pdeAD = self.net_fAD(self.x, self.y)
        loss_bc = torch.mean(f_bc ** 2)
        loss_f = torch.mean(f_pdeAD ** 2)
        loss = self.lbdbc * loss_bc + self.lbdf * loss_f

        if self.iter % 10 == 0:
            l2 = self.predict(x_area)[1]
            self.L2.append([time.perf_counter()-start, l2])

        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_f: %.5e, lbdbc: %.5e, lbdf: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_f.item(), self.lbdbc, self.lbdf)
            )
        self.iter += 1
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

    def loss_funcAD(self):
        self.optimizer_LBFGS.zero_grad()
        loss = self.Calculate_lossAD()
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
        # self.optimizer_LBFGS.step(self.loss_funcAD)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).double().to(device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).double().to(device)
        self.dnn.eval()
        u = self.net_u(x, y)
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        u = u.detach().cpu().numpy()
        L2 = np.linalg.norm(u_function(x[:, 0], y[:, 0]) - u[:, 0]) / np.linalg.norm(u_function(x[:, 0], y[:, 0]))
        return u,L2

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

# 定义边界点的数量
def bc_point(num_points):
    # 生成在 [-1, 1] x [-1, 1] 方形域边界上等距分布的点
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    # 生成边界上的点
    x_boundary = np.concatenate((x, x,-np.ones(num_points), np.ones(num_points)))
    y_boundary = np.concatenate((-np.ones(num_points), np.ones(num_points),y,y))
    # 使用column_stack将x和y坐标组合成点
    return np.column_stack((x_boundary, y_boundary))

# 定义要绘制的函数
def u_function(x, y):
    return (0.1 * np.sin(2 * np.pi * x) + np.tanh(10 * x)) * np.sin(2 * np.pi * y)

## 主函数
dx = dy = 0.05
dxdy = dx*dx
d = 2.0
N1 = 2
N2 = 20
B = np.random.normal(0, 2, (N1,N2))
# layers = [2*N2]+[40]+[1]
layers = [2]+[40]+[40]+[1]
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x_bc = bc_point(100)
u_bc = u_function(x_bc[:,0], x_bc[:,1]).reshape(-1, 1)

x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
h = dx*1.4
neighborhoods, distances, distance_vectors = find_neighborhood_points(x_area, 2.0*h)
Adam_iter = 1000
LBFGS_iter = 1000
order = 2
model = PhysicsInformedNN(x_area, neighborhoods, distances, distance_vectors,h,B,order,
                 x_bc,u_bc, dxdy, layers,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", end - start, 'seconds')
# u_pred = model.predict(x_area)[0]
# print("MSE:", np.mean(np.square(u_pred[:, 0] - u_function(x_area[:,0], x_area[:,1]))))
# print("L2 error:", np.linalg.norm(u_function(x_area[:,0], x_area[:,1]) - u_pred[:, 0])/np.linalg.norm(u_function(x_area[:,0], x_area[:,1])))
#
# df1 = pd.DataFrame(np.array(model.eigenvalues_iter).T)
# df2 = pd.DataFrame(np.array(model.L2))
# with pd.ExcelWriter('output.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='Sheet1', index=False)
#     df2.to_excel(writer, sheet_name='Sheet2', index=False)

# plt.figure(figsize=(6, 6))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=u_pred[:, 0],s=5, cmap='jet')
# plt.xlim(-1.1, 1.1)
# plt.ylim(-1.1, 1.1)
# plt.colorbar()
# plt.show()
# plt.figure(figsize=(6, 6))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=np.abs(u_pred[:, 0]-u_function(x_area[:,0], x_area[:,1])),s=5, cmap='jet')
# plt.colorbar()
# plt.xlim(-1.1, 1.1)
# plt.ylim(-1.1, 1.1)
# plt.show()
# plt.figure()
# plt.yscale('log')
# plt.xscale('log')
# plt.plot(sorted(model.eigenvalues_iter[0], reverse=True),ls="-",lw=2,label="SK")
# plt.plot(sorted(model.eigenvalues_iter[1], reverse=True),ls="-",lw=2,label="AD")
# plt.plot(sorted(model.eigenvalues_iter[2], reverse=True),ls="-",lw=2,label="it=2000")