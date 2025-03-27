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
    def __init__(self, x_area, neighborhoods, distances, distance_vectors, h, Re,B,order,
                 X_bc_inflow,u_bc_inflow,X_bc_outflow,X_bc_noslip, dxdy,
                 layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.neighborhoods = neighborhoods.to(device)
        self.distances = distances.to(device)
        self.distance_vectors = distance_vectors.to(device)
        self.kernel = sph_kernel(distances, h).to(device)
        self.RKPM_C = Compute_C(self.distance_vectors, self.kernel.unsqueeze(-1), dxdy,order).to(device)
        self.h = h
        self.lbdf1 = self.lbdf2 = self.lbdbc_out = 1.0
        self.lbdubc_in = self.lbdvbc_in = self.lbdf3 = 1.0
        self.lbdubc_noslip = self.lbdvbc_noslip = 1.0
        self.B = torch.tensor(B).double().to(device)
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
        X = torch.cat([torch.cos(X @ self.B),torch.sin(X @ self.B)], dim=1)
        u = self.dnn(X)[:,0:3]
        return u

    def net_f(self, x, y):
        uvp = self.net_u(x,y)
        u_d = R_compute_field_gradients(uvp[:, 0:1], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        v_d = R_compute_field_gradients(uvp[:, 1:2], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        p_d = R_compute_field_gradients(uvp[:, 2:3], self.kernel.unsqueeze(-1), self.neighborhoods, self.RKPM_C)
        e1 = u_d[:, 0:1] + v_d[:, 1:2]
        e2 = uvp[:, 0:1] * u_d[:, 0:1] + uvp[:, 1:2] * u_d[:, 1:2] + p_d[:, 0:1] - 1 / self.Re * (u_d[:, 2:3] + u_d[:, 4:5])
        e3 = uvp[:, 0:1] * v_d[:, 0:1] + uvp[:, 1:2] * v_d[:, 1:2] + p_d[:, 1:2] - 1 / self.Re * (v_d[:, 2:3] + v_d[:, 4:5])
        return e1,e2,e3

    def net_bcout(self, x, y):
        uvp = self.net_u(x, y)
        u_x = torch.autograd.grad(uvp[:, 0:1], x, grad_outputs=torch.ones_like(uvp[:, 0:1]), retain_graph=True, create_graph=True)[0]
        f = u_x/self.Re - uvp[:, 2:3]
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
        loss = self.lbdubc_in * loss_ubc_in + self.lbdvbc_in * loss_vbc_in + \
               self.lbdbc_out *loss_bc_out + self.lbdubc_noslip * loss_ubc_noslip + self.lbdvbc_noslip * loss_vbc_noslip + \
               self.lbdf1 * loss_fe1 + self.lbdf2 * loss_fe2 + self.lbdf3 * loss_fe3
        self.iter += 1
        # if self.iter % 1000 == 0 and self.iter < self.Adam_iter:
        #     g_ubc_in = self.Calculate_loss_gradient(loss_ubc_in)
        #     g_vbc_in = self.Calculate_loss_gradient(loss_vbc_in)
        #     g_bc_out = self.Calculate_loss_gradient(loss_bc_out)
        #     g_ubc_noslip = self.Calculate_loss_gradient(loss_ubc_noslip)
        #     g_vbc_noslip = self.Calculate_loss_gradient(loss_vbc_noslip)
        #     g_fe1 = self.Calculate_loss_gradient(loss_fe1)
        #     g_fe2 = self.Calculate_loss_gradient(loss_fe2)
        #     g_fe3 = self.Calculate_loss_gradient(loss_fe3)
        #     sum_dlg = g_ubc_in + g_vbc_in + g_bc_out + g_ubc_noslip + g_vbc_noslip + g_fe1 + g_fe2 + g_fe3
        #     self.lbdubc_in = 0.9 * self.lbdubc_in + 0.1 * sum_dlg / g_ubc_in
        #     self.lbdvbc_in = 0.9 * self.lbdvbc_in + 0.1 * sum_dlg / g_vbc_in
        #     self.lbdbc_out = 0.9 * self.lbdbc_out + 0.1 * sum_dlg / g_bc_out
        #     self.lbdubc_noslip = 0.9 * self.lbdubc_noslip +0.1 * sum_dlg / g_ubc_noslip
        #     self.lbdvbc_noslip = 0.9 * self.lbdvbc_noslip +0.1 * sum_dlg / g_vbc_noslip
        #     self.lbdf1 = 0.9 * self.lbdf1 +0.1 * sum_dlg / g_fe1
        #     self.lbdf2 = 0.9 * self.lbdf2 +0.1 * sum_dlg / g_fe2
        #     self.lbdf3 =0.9 * self.lbdf3 +0.1 * sum_dlg / g_fe3
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, loss_ubc_in: %.5e, loss_vbc_in: %.5e, loss_bc_out: %.5e,loss_ubc_noslip: %.5e, loss_vbc_noslip: %.5e, loss_fe1: %.5e, Loss_fe2: %.5e, Loss_fe3: %.5e\n'
                'lbdubc_in: %.5e, lbdvbc_in: %.5e,lbdvbc_in: %.5e,lbdbc_out: %.5e, lbdvbc_noslip: %.5e, lbdf1: %.5e, lbdf2: %.5e, lbdf3: %.5e, '
                % (self.iter, loss.item(), loss_ubc_in.item(), loss_vbc_in.item(), loss_bc_out.item(), loss_ubc_noslip.item(), loss_vbc_noslip.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item(),
                   self.lbdubc_in, self.lbdvbc_in,self.lbdbc_out,self.lbdubc_noslip, self.lbdvbc_noslip, self.lbdf1, self.lbdf2, self.lbdf3)
            )
        self.loss.append([self.iter, loss.item(), loss_ubc_in.item(), loss_vbc_in.item(),loss_ubc_noslip.item(), loss_vbc_noslip.item(), loss_fe1.item(), loss_fe2.item(), loss_fe3.item()])
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
        p = self.net_u(x, y)[:, 2:3]
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        p = p.detach().cpu().numpy()
        return u, v, p


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
    q = distances / h[:, None]  # 将 h 扩展为形状为 (**, 1) 的列向量，然后进行除法
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

# 主函数
Re = 300
N1 = 2
N2 = 64
B = np.random.normal(0, 5, (N1,N2))
layers = [2*N2]+[20]*3
layers_u =[20]*3+[1]
layers_v =[20]*3+[1]
layers_p = [20]*3+[1]
ub = np.array([2,1])
lb = np.array([0,0])

# 从Excel加载数据到DataFrame
data = pd.read_excel('coord2.xlsx')
# 将DataFrame转换为NumPy数组
x_area = data.to_numpy()

# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c='black', marker='.',s=6)
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.show()

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
order = 2
neighborhoods, distances, distance_vectors, neighborhood_radius = find_neighborhood_points(x_area,k_neighbors=21)
h = neighborhood_radius*1.1/2.0
model = PhysicsInformedNN(x_area, neighborhoods, distances, distance_vectors, h, Re,B,order,
                          X_bc_inflow,u_bc_inflow,X_bc_outflow,X_bc_noslip, dxdy,
                          layers,layers_u,layers_v,layers_p,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", end - start, 'seconds')

dx = dy = 0.001
x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
u_pred,v_pred,p_pred = model.predict(x_area)
u_pred = u_pred.reshape(X.shape[0], X.shape[1])
v_pred = v_pred.reshape(X.shape[0], X.shape[1])
p_pred = p_pred.reshape(X.shape[0], X.shape[1])
# Define the circular region to exclude
center_x, center_y = 0.5, 0.5
radius = 0.1
mask = (X - center_x)**2 + (Y - center_y)**2 < radius**2
# Set the velocities inside the circle to zero
u_pred[mask] = np.nan
v_pred[mask] = np.nan
p_pred[mask] = np.nan
fig = plt.figure(figsize = (12, 5))
plt.streamplot(X, Y, u_pred, v_pred, density = 1.8,arrowsize=0,integration_direction='both',color='black',linewidth=0.5)

# u_pred,v_pred,p_pred = model.predict(x_area)
u_bc,v_bc,p_bc = model.predict(bc_cylinder)
x_ = np.column_stack((np.linspace(0.75, 0.75, 1000), np.linspace(0.0, 1.0, 1000)))
u_,v_,p_ = model.predict(x_)

# df1 = pd.DataFrame(np.vstack((x_[:, 1],u_[:, 0])).T)
# df2 = pd.DataFrame(np.vstack((x_[:, 1],v_[:, 0])).T)
# df3 = pd.DataFrame(np.vstack((jiaodu*r,p_bc[:, 0])).T)
# with pd.ExcelWriter('dataSK.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='Sheet1', index=False)
#     df2.to_excel(writer, sheet_name='Sheet2', index=False)
#     df3.to_excel(writer, sheet_name='Sheet3', index=False)

V_ = (u_**2+v_**2)**0.5
V_P = (u_pred**2+v_pred**2)**0.5

cp = plt.pcolor(X, Y, V_P, cmap='jet')  # 绘制填充等高线图
plt.xlim(0, 2)
plt.ylim(0, 1)
plt.colorbar()
plt.show()

# plt.figure(figsize=(12, 5))
# cp = plt.pcolor(X, Y, p_pred, cmap='jet')  # 绘制填充等高线图
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()


# # 从Excel加载数据到DataFrame
# data = pd.read_excel('comsol.xlsx')
# # 将DataFrame转换为NumPy数组
# Comsoldata = data.to_numpy()
#
# plt.figure(figsize=(12, 5))
# plt.scatter(Comsoldata[:, 0], Comsoldata[:, 1],c=Comsoldata[:, 2],s=5, cmap='jet')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()

# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=V_P[:, 0],s=3, cmap='jet')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()
# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=u_pred[:, 0],s=3, cmap='jet')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()
# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=v_pred[:, 0],s=2, cmap='jet')
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.colorbar()
# plt.show()
# #
# plt.figure()
# plt.plot(jiaodu*r, p_bc[:, 0])
# plt.show()
# #
# plt.figure()
# plt.plot(x_[:, 1], v_[:, 0])
# plt.show()
# # #
# plt.figure(figsize=(12, 5))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=p_pred[:, 0],s=3, cmap='jet')
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
#
# plt.figure(figsize=(24, 8))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=V_Ture[:, 0]-V_P[:, 0],s=5, cmap='jet')
# plt.xlim(0, 2.2)
# plt.ylim(0, 0.41)
# plt.colorbar()
# plt.show()