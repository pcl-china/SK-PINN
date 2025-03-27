from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, x_area, x_bc,u_bc, layers,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
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
        u = self.dnn(X)[:,0:1]
        return u

    def net_f(self, x, y):
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y,grad_outputs=torch.ones_like(u_y),retain_graph=True,create_graph=True)[0]
        q = torch.sin(2 * np.pi * y) * (20 * torch.tanh(10 * x) * (10 * torch.tanh(10 * x) ** 2 - 10) - (
                    2 * (np.pi) ** 2 * torch.sin(2 * np.pi * x)) * 0.2) - 4 * (np.pi) ** 2 * torch.sin(
            2 * np.pi * y) * (torch.tanh(10 * x) + torch.sin(2 * np.pi * x) * 0.1)
        f = u_xx+u_yy-q
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
        f_pde = self.net_f(self.x, self.y)
        # f = torch.cat((f_pde, f_bc), dim=0)
        # eigenvalues = self.NTK(f)
        loss_bc = torch.mean((f_bc) ** 2)
        loss_f = torch.mean(f_pde ** 2)
        loss = 1 * loss_bc + 1 * loss_f
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_f: %.5e' % (
                    self.iter, loss.item(), loss_bc.item(), loss_f.item())
            )
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_f.item()])
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
        u = self.net_u(x, y)
        u_x = torch.autograd.grad(u, x,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_y = torch.autograd.grad(u, y,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        f_pde = self.net_f(x, y)
        u = u.detach().cpu().numpy()
        u_x = u_x.detach().cpu().numpy()
        u_y = u_y.detach().cpu().numpy()
        f_pde = f_pde.detach().cpu().numpy()
        return u, u_x, u_y, f_pde

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
d = 2.0
layers = [2]+[40]+[40]+[1]
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x_bc = bc_point(100)
u_bc = u_function(x_bc[:,0], x_bc[:,1]).reshape(-1, 1)


x = np.arange(lb[0]-dx/2,ub[0]+dx/2,dx)
y = np.arange(lb[1]-dy/2,ub[1]+dy/2,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
Adam_iter = 1000
LBFGS_iter = 1000
model = PhysicsInformedNN(x_area, x_bc,u_bc, layers, ub, lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", end - start, 'seconds')
u_pred=model.predict(x_area)[0]
print("MSE:", np.mean(np.square(u_pred[:, 0] - u_function(x_area[:,0], x_area[:,1]))))
print("L2 error:", np.linalg.norm(u_function(x_area[:,0], x_area[:,1]) - u_pred[:, 0])/np.linalg.norm(u_function(x_area[:,0], x_area[:,1])))

plt.figure(figsize=(6, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=u_pred[:, 0],s=5, cmap='jet')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.colorbar()
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=np.abs(u_pred[:, 0]-u_function(x_area[:,0], x_area[:,1])),s=5, cmap='jet')
plt.colorbar()
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()


# loss曲线
# plt.figure()
# plt.yscale('log')
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,1],ls="-",lw=2,label="loss")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,2],ls="-",lw=2,label="loss_bc")
# plt.plot(np.array(model.loss)[:,0],np.array(model.loss)[:,3],ls="-",lw=2,label="loss_f")
# plt.legend()
# plt.grid(linestyle=":")
# # plt.axvline(x=1000,c="b",ls="--",lw=2)
# plt.xlim(0,np.array(model.loss)[-1,0])
# plt.show()