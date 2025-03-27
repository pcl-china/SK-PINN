from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence
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
                 x_bc,n_bc, dxdy, layers,ub,lb,Adam_iter,LBFGS_iter):
        self.x = torch.tensor(x_area[:, 0:1], requires_grad=True).double().to(device)
        self.y = torch.tensor(x_area[:, 1:2], requires_grad=True).double().to(device)
        self.neighborhoods = neighborhoods.to(device)
        self.distances = distances.to(device)
        self.distance_vectors = distance_vectors.to(device)
        self.kernel = sph_kernel(distances, h).to(device)
        self.RKPM_C = Compute_C(self.distance_vectors, self.kernel.unsqueeze(-1), dxdy,order).to(device)
        self.h = h
        self.lbdbc = self.lbddbc = self.lbdf = 1.0
        self.B = torch.tensor(B).double().to(device)
        self.x_bc = torch.tensor(x_bc[:, 0:1], requires_grad=True).double().to(device)
        self.y_bc = torch.tensor(x_bc[:, 1:2], requires_grad=True).double().to(device)
        self.n_bcx = torch.tensor(n_bc[:, 0:1]).double().to(device)
        self.n_bcy = torch.tensor(n_bc[:, 1:2]).double().to(device)
        self.ub = torch.tensor(ub).double().to(device)
        self.lb = torch.tensor(lb).double().to(device)
        self.dxdy = dxdy
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
        u = self.net_u(x,y)
        u_d = R_compute_field_gradients(u,self.kernel.unsqueeze(-1), self.neighborhoods,self.RKPM_C)
        f =u_d[:,0:1]+2*u_d[:,1:2]+u_d[:,2:3] - 10
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

        loss = self.lbdbc * loss_bc + self.lbddbc * loss_dbc + self.lbdf * loss_f
        self.iter += 1
        if self.iter % 1000 == 0 and self.iter < self.Adam_iter:
            g_ubc = self.Calculate_loss_gradient(loss_bc)
            g_vbc = self.Calculate_loss_gradient(loss_dbc)
            g_fe = self.Calculate_loss_gradient(loss_f)
            sum_dlg = g_ubc + g_vbc + g_fe
            self.lbdbc = 0.9 * self.lbdbc +0.1 * sum_dlg / g_ubc
            self.lbddbc = 0.9 * self.lbddbc +0.1 * sum_dlg / g_vbc
            self.lbdf = 0.9 * self.lbdf +0.1 * sum_dlg / g_fe
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_bc: %.5e, Loss_dbc: %.5e, Loss_f1: %.5e\n'
                'lbdbc: %.5e, lbddbc: %.5e, lbdf: %.5e '
                % (self.iter, loss.item(), loss_bc.item(), loss_dbc.item(),loss_f.item(),
                   self.lbdbc, self.lbddbc, self.lbdf)
            )
        self.loss.append([self.iter, loss.item(), loss_bc.item(), loss_dbc.item(), loss_f.item()])
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
        # u_d = R_compute_field_gradients(u,self.kernel.unsqueeze(-1), self.neighborhoods,self.RKPM_C)
        # F_xx = u_d[:, 0:1] + 0.3 * u_d[:, 2:3]
        # F_yy = u_d[:, 2:3] + 0.3 * u_d[:, 0:1]
        # F_xy = u_d[:, 1:2]*0.0
        # mises = ((F_xx-F_yy)**2+F_xy**2)**.5
        # f_pde = self.net_f(x, y)
        u = u.detach().cpu().numpy()
        # mises = mises.detach().cpu().numpy()
        # f_pde = f_pde.detach().cpu().numpy()
        return u

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
    q = distances / h[:, None]  # 将 h 扩展为形状为 (13595, 1) 的列向量，然后进行除法
    result = torch.zeros_like(distances, dtype=torch.float64)
    # 计算核函数的值
    within_range = (0 <= q) & (q <= 2)
    result[within_range] = (
        (2/3 - 1.0 * q[within_range] ** 2 + 0.5 * q[within_range] ** 3) * (q[within_range] <= 1) +
        (1/6 * (2 - q[within_range]) ** 3) * ((1 < q[within_range]) & (q[within_range] <= 2))
    )
    result = (15 / (7 * np.pi * h[:, None] ** 2)) * result
    return result

# # 有限核粒子方法
# # 先需要根据distance_vectors（维度N,n,2）组一个向量列表（维度N,n,最后一个维度根据需求调整），然后做求逆操作
# def FPM_Matrix_inverse(kernel,kernel_GradVector,distance_vectors,distances,dxdy):
#     Moment_Vector1 = torch.cat([torch.ones(distances.shape).unsqueeze(-1).to(device),distance_vectors], dim=2)
#     kernel_Vector = torch.cat([kernel.unsqueeze(-1) * dxdy,kernel_GradVector], dim=2)
#     Matrix = torch.matmul(kernel_Vector.unsqueeze(3), Moment_Vector1.unsqueeze(2))
#     Matrix_sum = torch.sum(Matrix, dim=1)
#     Matrix_inverse = torch.inverse(Matrix_sum)
#     return Matrix_inverse,kernel_Vector
#
# def FPM_compute_field_gradients(f_values,kernel_Vector, neighborhoods,Matrix_inverse):
#     u_ngr = f_values[neighborhoods][:,:,0]*(neighborhoods != -1)
#     # 将邻域内点的梯度值按照核函数梯度进行加权求和
#     field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',kernel_Vector,(u_ngr-f_values)),dim = 1)
#     FPM_field_gradients = torch.matmul(Matrix_inverse[:,1:3,:], field_gradients.unsqueeze(-1)).squeeze(-1)
#     return FPM_field_gradients

# 再生核粒子方法
def Compute_C(distance_vectors, kernel, dxdy, order):
    moment_terms = [torch.ones(kernel.shape).to(device)]
    terms_num = np.sum(np.arange(1, order + 2))
    for i in range(1, order + 1):
        for j in range(i + 1):
            term = (distance_vectors[:, :, 0:1] ** (i - j)) * (distance_vectors[:, :, 1:2] ** j) / (dxdy ** (i / 2))
            moment_terms.append(term)
    moment_vector = torch.cat(moment_terms, dim=2)
    H = torch.tensor([[0,0,0,0,0,0,0,0,0,0,24/dxdy ** 2,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,4/dxdy ** 2,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,24/dxdy ** 2]], dtype=torch.double)
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
dx = dy = 0.01
dxdy = 1.0
d = 2.0
N1 = 2
N2 = 20
B = np.random.normal(0, 1, (N1,N2))
layers = [2*N2]+[40]+[40]+[1]
ub = np.array([d/2,d/2])
lb = np.array([-d/2,-d/2])
x_bc = generate_boundary_points(100)
n_out = find_normal(x_bc)

x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
neighborhoods, distances, distance_vectors, neighborhood_radius = find_neighborhood_points(x_area,k_neighbors=45)
h = neighborhood_radius*1.1/2.0
Adam_iter = 10000
LBFGS_iter = 10000
order = 5
model = PhysicsInformedNN(x_area, neighborhoods, distances, distance_vectors,h,B,order,
                 x_bc,n_out, dxdy, layers,ub,lb,Adam_iter,LBFGS_iter)
start = time.perf_counter()
model.train()
end = time.perf_counter()
print("训练时间为", round(end - start), 'seconds')
u_pred = model.predict(x_area)
x_ = np.column_stack((np.linspace(-1.0, 1.0, 1000), np.linspace(0.0, 0.0, 1000)))
u_ = model.predict(x_)

df1 = pd.DataFrame(np.vstack((x_[:, 0],u_[:, 0])).T)
df1.to_excel('data.xlsx', sheet_name='Sheet1', index=False)

plt.figure(figsize=(6, 6))
plt.scatter(x_area[:, 0], x_area[:, 1],c=u_pred[:, 0],s=5, cmap='jet')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.colorbar()
plt.show()

# plt.figure(figsize=(6, 6))
# plt.scatter(x_area[:, 0], x_area[:, 1],c=mises[:, 0],s=5, cmap='jet')
# plt.xlim(-1.1, 1.1)
# plt.ylim(-1.1, 1.1)
# plt.colorbar()
# plt.show()