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
    def __init__(self, x_area,x_bc,n_bc, layers,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.n_bcx = torch.tensor(n_bc[:, 0:1]).double().to(device)
        self.n_bcy = torch.tensor(n_bc[:, 1:2]).double().to(device)
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
        u = self.dnn(X)[:,0:1]
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_xxx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]
        u_xxxx = torch.autograd.grad(u_xxx, x, grad_outputs=torch.ones_like(u_xxx), retain_graph=True, create_graph=True)[0]
        u_xxy = torch.autograd.grad(u_xx, y, grad_outputs=torch.ones_like(u_xx), retain_graph=True, create_graph=True)[0]
        u_xxyy = torch.autograd.grad(u_xxy, y, grad_outputs=torch.ones_like(u_xxy), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        u_yyy = torch.autograd.grad(u_yy, y, grad_outputs=torch.ones_like(u_yy), retain_graph=True, create_graph=True)[0]
        u_yyyy = torch.autograd.grad(u_yyy, y, grad_outputs=torch.ones_like(u_yyy), retain_graph=True, create_graph=True)[0]

        f = u_xxxx+2*u_xxyy+u_yyyy - 10
        return f

    def net_bc(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        dbc = u_x*self.n_bcx+u_y*self.n_bcy
        return u,dbc

    def Calculate_loss(self):
        bc,dbc = self.net_bc(self.x_bc, self.y_bc)
        f_pde = self.net_f(self.x, self.y)
        loss_bc = torch.mean(bc ** 2)
        loss_dbc = torch.mean(dbc ** 2)
        loss_f = torch.mean(f_pde ** 2)
        self.iter += 1

        loss = 1.0 * loss_bc + 1.0 * loss_dbc + 1.0 * loss_f

        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_dbc: %.5e, Loss_f: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_dbc.item(),loss_f.item())
            )
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_dbc.item(), loss_f.item()])
        return loss

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

def generate_boundary_points(num_points_per_side):
    # 生成方形边界的四个顶点
    square_points = np.array([
        [-1, -1],  # 左下角
        [-1, 1],   # 左上角
        [1, 1],    # 右上角
        [1, -1]    # 右下角
    ])
    # 每个边的点数
    num_points = num_points_per_side
    # 按逆时针顺序连接边界点
    left_points = np.linspace(square_points[0], square_points[1], num_points, endpoint=False)
    top_points = np.linspace(square_points[1], square_points[2], num_points, endpoint=False)
    right_points = np.linspace(square_points[2], square_points[3], num_points, endpoint=False)
    bottom_points = np.linspace(square_points[3], square_points[0], num_points, endpoint=False)
    # 将所有边界点连接起来
    boundary_points = np.vstack((left_points, top_points, right_points, bottom_points))
    return boundary_points
## 求单位法向
def find_normal(x):
    n_unit = np.array(x, copy=True)
    for i in range(x.shape[0]):
        a = x[i, :] - x[i - 1, :]
        b = x[(i + 1) * (i != x.shape[0] - 1), :] - x[i, :]
        a_n = np.array(a, copy=True)
        a_n[0], a_n[1] = a[1], -a[0]
        b_n = np.array(b, copy=True)
        b_n[0], b_n[1] = b[1], -b[0]
        a_abs = (a_n[0] ** 2 + a_n[1] ** 2) ** 0.5
        b_abs = (b_n[0] ** 2 + b_n[1] ** 2) ** 0.5
        n = b_n / b_abs + a_n / a_abs
        n_abs = (n[0] ** 2 + n[1] ** 2) ** 0.5
        n_unit[i, :] = n / n_abs
    return n_unit

## 主函数
dx = dy = 0.02
dxdy = dx*dy
d = 2.0
layers = [2]+[40]+[40]+[40]+[1]
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x_bc = generate_boundary_points(100)
n_out = find_normal(x_bc)

x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
Adam_iter = 10000
LBFGS_iter = 10000
model = PhysicsInformedNN(x_area,x_bc,n_out, layers,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')

u_pred = model.predict(x_area)[0]
plt.figure(figsize=(6, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=u_pred[:, 0],s=5, cmap='jet')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.colorbar()
plt.show()