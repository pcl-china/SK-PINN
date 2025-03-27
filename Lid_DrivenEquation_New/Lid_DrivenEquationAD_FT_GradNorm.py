from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

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
    def __init__(self, x_area, x_bc,U_bc,Re,B,
                 layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.u_bc = torch.tensor(U_bc[:, 0:1]).double().to(device)
        self.v_bc = torch.tensor(U_bc[:, 1:2]).double().to(device)
        self.Re = torch.tensor(Re).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.lbdf1 = self.lbdf2 = self.lbdf3 = 1.0
        self.lbdubc = self.lbdvbc = 1.0
        self.B = torch.tensor(B).double().to(device)
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

    def Calculate_loss(self):
        bc = self.net_u(self.x_bc, self.y_bc)[:,0:2]
        f_e1,f_e2,f_e3 = self.net_f(torch.cat((self.x, self.x_bc), dim=0), torch.cat((self.y, self.y_bc), dim=0))
        loss_ubc = torch.mean((bc[:,0:1] - self.u_bc) ** 2)
        loss_vbc = torch.mean((bc[:,1:2] - self.v_bc) ** 2)
        loss_fe1 = torch.mean(f_e1 ** 2)
        loss_fe2 = torch.mean(f_e2 ** 2)
        loss_fe3 = torch.mean(f_e3 ** 2)
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
        if self.iter % 100 == 0:
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

## 主函数
dx = dy = 0.004
d = 1.0
Re = 400
N1 = 2
N2 = 128
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
Adam_iter = 2000
LBFGS_iter = 4000
model = PhysicsInformedNN(x_area,x_bc,u_bc,Re,B,
                 layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter)

start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')

mat = pd.read_excel('Re400.xlsx').to_numpy()
V_Ture = mat[:,2:3]
X_area = np.hstack((mat[:,0:1],mat[:,1:2]))
u_pred,v_pred = model.predict(X_area)
V_P = (u_pred**2+v_pred**2)**0.5
# np.save('V_PmAD.npy', V_P)
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
# plt.scatter(X_area[:,0], X_area[:,1], c = np.abs(V_Ture[:,0]-V_P[:,0]),s=0.6, cmap='jet',vmax=0.05,vmin=0.0)
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