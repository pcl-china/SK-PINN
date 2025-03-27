from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
import time

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
        q = torch.sin(2 * np.pi * y) * (20 * torch.tanh(10 * x) * (10 * torch.tanh(10 * x) ** 2 - 10) - (
                2 * (np.pi) ** 2 * torch.sin(2 * np.pi * x)) * 0.2) - 4 * (np.pi) ** 2 * torch.sin(
            2 * np.pi * y) * (torch.tanh(10 * x) + torch.sin(2 * np.pi * x) * 0.1)
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
dx = dy = 0.04
dxdy = dx*dy
d = 2.0
N1 = 2
N2 = 20
B = np.random.normal(0, 2, (N1,N2))
layers = [2*N2]+[40]+[1]
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x_bc = bc_point(100)
u_bc = u_function(x_bc[:,0], x_bc[:,1]).reshape(-1, 1)

x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
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
# plt.figure(figsize=(6, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=u_pred[:, 0],s=5, cmap='jet')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.colorbar()
plt.show()
# plt.figure(figsize=(6, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=np.abs(u_pred[:, 0]-u_function(x_area[:,0], x_area[:,1])),s=5, cmap='jet')
plt.colorbar()
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()