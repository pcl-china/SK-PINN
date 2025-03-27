from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import scipy.io
import time
import math
import shapely.geometry

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
    def __init__(self, x_area,X_bc_inflow,u_bc_inflow,X_bc_outflow,X_bc_noslip,Re,
                 layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.x_bcin = torch.tensor(X_bc_inflow[:, 0:1], requires_grad=True).double().to(device)
        self.y_bcin = torch.tensor(X_bc_inflow[:, 1:2], requires_grad=True).double().to(device)
        self.u_bcin = torch.tensor(u_bc_inflow.reshape(-1,1)).double().to(device)
        self.x_bcout = torch.tensor(X_bc_outflow[:, 0:1], requires_grad=True).double().to(device)
        self.y_bcout = torch.tensor(X_bc_outflow[:, 1:2], requires_grad=True).double().to(device)
        self.x_bcnoslip = torch.tensor(X_bc_noslip[:, 0:1], requires_grad=True).double().to(device)
        self.y_bcnoslip = torch.tensor(X_bc_noslip[:, 1:2], requires_grad=True).double().to(device)
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
        u = self.dnn(X)[:,0:3]
        return u

    def net_f(self, x, y):
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

    def net_bcout(self, x, y):
        uvp = self.net_u(x, y)
        u_x = torch.autograd.grad(uvp[:, 0:1], x, grad_outputs=torch.ones_like(uvp[:, 0:1]), retain_graph=True, create_graph=True)[0]
        f = u_x - uvp[:, 2:3]
        return f

    def Calculate_loss(self):
        bc_in = self.net_u(self.x_bcin, self.y_bcin)[:,0:2]
        bc_noslip = self.net_u(self.x_bcnoslip, self.y_bcnoslip)[:, 0:2]
        f_e1,f_e2,f_e3 = self.net_f(self.x, self.y)
        loss_ubc_in = torch.mean((bc_in[:,0:1]-self.u_bcin) ** 2)
        loss_vbc_in = torch.mean((bc_in[:,1:2]) ** 2)
        loss_bc_out = torch.mean((self.net_bcout(self.x_bcout, self.y_bcout)) ** 2)
        loss_ubc_noslip = torch.mean((bc_noslip[:, 0:1]) ** 2)
        loss_vbc_noslip = torch.mean((bc_noslip[:, 1:2]) ** 2)
        loss_fe1 = torch.mean(f_e1 ** 2)
        loss_fe2 = torch.mean(f_e2 ** 2)
        loss_fe3 = torch.mean(f_e3 ** 2)
        loss = 1.0 * loss_ubc_in + 1.0 * loss_vbc_in + 1.0 *loss_bc_out + 1.0 * loss_ubc_noslip + 1.0 * loss_vbc_noslip + \
               1.0 * loss_fe1 + 1.0 * loss_fe2 + 1.0 * loss_fe3
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, loss_ubc_in: %.5e, loss_vbc_in: %.5e, loss_bc_out: %.5e,loss_ubc_noslip: %.5e, loss_vbc_noslip: %.5e, loss_fe1: %.5e, Loss_fe2: %.5e, Loss_fe3: %.5e'
                % (self.iter, loss.item(), loss_ubc_in.item(), loss_vbc_in.item(), loss_bc_out.item(), loss_ubc_noslip.item(), loss_vbc_noslip.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item())
            )
        self.loss.append([self.iter, loss.item(), loss_ubc_in.item(), loss_vbc_in.item(),loss_ubc_noslip.item(), loss_vbc_noslip.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item()])
        return loss

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
        p = self.net_u(x, y)[:, 2:3]
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        return u, v, p

def bcnoslip_point(ub,lb,Distance_points,bc_cylinder):
    # 生成在 [-1, 1] x [-1, 1] 方形域边界上等距分布的点
    x = np.arange(lb[0] - 0.0 * Distance_points, ub[0] + 1.0 * Distance_points, Distance_points)
    y = np.ones_like(x)
    # 生成边界上的点
    x_boundary = np.concatenate((x, x))
    y_boundary = np.concatenate((lb[1]*y, ub[1]*y))
    wallbc = np.column_stack((x_boundary, y_boundary))
    bc = np.vstack((wallbc, bc_cylinder))
    return bc

# 主函数
Re = 300
layers = [2]+[128]+[20]*3
layers_u =[20]*3+[1]
layers_v =[20]*3+[1]
layers_p = [20]*3+[1]
ub = np.array([2,1])
lb = np.array([0,0])

# 从Excel加载数据到DataFrame
data = pd.read_excel('coord2.xlsx')
# 将DataFrame转换为NumPy数组
x_area = data.to_numpy()

dx = dy = 0.004
dxdy = 1.0
r = 0.1
jiaodu = np.arange(0,2*math.pi,dx/r)
bc_cylinder = np.vstack((0.5+r*np.cos(jiaodu+math.pi), 0.5+r*np.sin(jiaodu+math.pi))).T
x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X_bc_noslip =bcnoslip_point(ub,lb,dx,bc_cylinder)
X_bc_inflow = np.column_stack((lb[0]*np.ones_like(y), y))
u_bc_inflow = 1.0 * 4 * X_bc_inflow[:,1] * (ub[1] - X_bc_inflow[:,1]) / (ub[1]**2)
X_bc_outflow = np.column_stack((ub[0]*np.ones_like(y), y))

Adam_iter = 20000
LBFGS_iter = 40000
model = PhysicsInformedNN(x_area,X_bc_inflow,u_bc_inflow,X_bc_outflow,X_bc_noslip,Re,
                          layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", end - start, 'seconds')

u_pred,v_pred,p_pred = model.predict(x_area)
u_bc,v_bc,p_bc = model.predict(bc_cylinder)
x_ = np.column_stack((np.linspace(0.75, 0.75, 1000), np.linspace(0.0, 1.0, 1000)))
u_,v_,p_ = model.predict(x_)

# df1 = pd.DataFrame(np.vstack((x_[:, 1],u_[:, 0])).T)
# df2 = pd.DataFrame(np.vstack((x_[:, 1],v_[:, 0])).T)
# df3 = pd.DataFrame(np.vstack((jiaodu*r,p_bc[:, 0])).T)
# with pd.ExcelWriter('dataAD.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='Sheet1', index=False)
#     df2.to_excel(writer, sheet_name='Sheet2', index=False)
#     df3.to_excel(writer, sheet_name='Sheet3', index=False)


# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=V_P[:, 0]*0.3,s=2, cmap='jet')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()
# #
# plt.figure()
# plt.plot(bc_cylinder[:, 1], p_bc[:, 0]*0.09)
# plt.show()
# #
# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=p_pred[:, 0]*0.09,s=2, cmap='jet')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()

# plt.figure(figsize=(24, 8))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=V_Ture[:, 0],s=5, cmap='jet')
# plt.xlim(0, 2.2)
# plt.ylim(0, 0.41)
# plt.colorbar()
# plt.show()

# plt.figure(figsize=(24, 8))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=V_Ture[:, 0]-V_P[:, 0],s=5, cmap='jet')
# plt.xlim(0, 2.2)
# plt.ylim(0, 0.41)
# plt.colorbar()
# plt.show()