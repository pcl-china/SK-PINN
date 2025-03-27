import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence

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

# 再生核粒子方法
def Compute_C(distance_vectors, kernel, dxdy, order):
    moment_terms = [torch.ones(kernel.shape)]
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
    H0 = torch.nn.functional.pad(H, (0, terms_num-H.shape[1]), value=0)
    matrix = torch.matmul(moment_vector.unsqueeze(3), moment_vector.unsqueeze(2)) * kernel.unsqueeze(-1)
    matrix_sum = torch.sum(matrix, dim=1)
    matrix_inverse = torch.inverse(matrix_sum)
    C = torch.matmul(torch.matmul(moment_vector, matrix_inverse), H0.t().view(1, 1, terms_num, -1)).squeeze(0)
    return C

def Compute_C2(distance_vectors, kernel, dxdy, order):
    moment_terms = [torch.ones(kernel.shape)]
    terms_num = np.sum(np.arange(1, order + 2))
    for i in range(1, order + 1):
        for j in range(i + 1):
            term = (distance_vectors[:, :, 0:1] ** (i - j)) * (distance_vectors[:, :, 1:2] ** j) / (dxdy ** (i / 2))
            moment_terms.append(term)
    moment_vector = torch.cat(moment_terms, dim=2)
    H = torch.tensor([[0,0,0,0,0,0,0,0,0,0,24/dxdy ** 2,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,4/dxdy ** 2,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,24/dxdy ** 2]], dtype=torch.double)
    H0 = torch.nn.functional.pad(H, (0, terms_num-H.shape[1]), value=0)
    matrix = torch.matmul(moment_vector.unsqueeze(3), moment_vector.unsqueeze(2)) * kernel.unsqueeze(-1)
    matrix_sum = torch.sum(matrix, dim=1)
    matrix_inverse = torch.inverse(matrix_sum)
    C = torch.matmul(torch.matmul(moment_vector, matrix_inverse), H0.t().view(1, 1, terms_num, -1)).squeeze(0)
    return C

def R_compute_field_gradients(f_values,kernel, neighborhoods, distances, distance_vectors,dxdy,C):
    u_ngr = f_values[neighborhoods][:,:,0]*(neighborhoods != -1)
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(u_ngr.unsqueeze(-1)*C*kernel,dim = 1)
    return field_gradients


def f(x, y):
    # return torch.abs(x+y)
    return torch.sin(10*x)*torch.cos(10*y)
def f_dx(x, y):
    return 10 * torch.cos(10*x)*torch.cos(10*y)
def f_dxy(x, y):
    return - 100 * torch.cos(10*x)*torch.sin(10*y)
# def f_dy(x, y):
    # return - 10 * torch.sin(10*x)*torch.sin(10*y)
# def f_dxx(x, y):
#     return - 100 * torch.sin(10*x)*torch.cos(10*y)
# def f_dyy(x, y):
    # return - 100 * torch.sin(10*x)*torch.cos(10*y)

def f_dxxyy(x, y):
    return 10000 * torch.sin(10*x)*torch.cos(10*y)

# ## 主函数
# dx = dy = 0.02
# dxdy = dx*dy
# ub = np.array([1,1])
# lb = np.array([-1,-1])
# x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
# y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
# X,Y = np.meshgrid(x,y)
# x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
# x_area_tensor = torch.from_numpy(x_area)
#
# # num_points = 20000  # 你想要的点的数量
# # ub = np.array([1,1])
# # lb = np.array([-1,-1])
# # dxdy = (ub[0]-lb[0])*(ub[1]-lb[1])/num_points
# # # 生成随机坐标
# # x = np.random.uniform(lb[0], ub[0], num_points)
# # y = np.random.uniform(lb[1], ub[1], num_points)
# # x_area = np.column_stack((x, y))
# # # 将坐标转换为张量
# # x_area_tensor = torch.tensor(x_area, dtype=torch.double)
#
#
#
#
#
# u = f(x_area_tensor[:,0:1], x_area_tensor[:,1:2])
#
# neighborhoods, distances, distance_vectors, neighborhood_radius = find_neighborhood_points(x_area, k_neighbors=29)
# h = neighborhood_radius*1.1/2.0
#
# kernel = sph_kernel(distances, h)
#
# RKPM_C2 = Compute_C(distance_vectors,kernel.unsqueeze(-1),dxdy,2)
# f_d_RKPM2 = R_compute_field_gradients(u,kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C2)
# f_d_RKPM3 = R_compute_field_gradients(f_d_RKPM2[:,3:4],kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C2)
#
# RKPM_C4 = Compute_C2(distance_vectors,kernel.unsqueeze(-1),dxdy,4)
# f_d_RKPM4 = R_compute_field_gradients(u,kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C4)
#
# RKPM_C5 = Compute_C2(distance_vectors,kernel.unsqueeze(-1),dxdy,5)
# f_d_RKPM5 = R_compute_field_gradients(u,kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C5)
#
#
#
# fig = plt.figure()
# # ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
# plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = f_dxxyy(x_area_tensor[:,0:1], x_area_tensor[:,1:2]),s=5, cmap='jet')
# plt.colorbar()
# plt.show()
#
# # u_values = (f_d_RKPM2[:,3:4])
# # fig = plt.figure()
# # # ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
# # plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
# # plt.colorbar()
# # plt.show()
# # print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
#
# u_values = (f_d_RKPM3[:,3:4])
# fig = plt.figure()
# # ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
# plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxxyy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=5000.0,vmin=0.0)
# plt.colorbar()
# plt.show()
# print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
#
# u_values = (f_d_RKPM4[:,1:2])
# fig = plt.figure()
# # ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
# plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxxyy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=5000.0,vmin=0.0)
# plt.colorbar()
# plt.show()
# print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
#
# u_values = (f_d_RKPM5[:,1:2])
# fig = plt.figure()
# # ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
# plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxxyy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=5000.0,vmin=0.0)
# plt.colorbar()
# plt.show()
# print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())


# # 计算 u 值
# u_values = u.reshape(X.shape)
# # 创建 3D 图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # 绘制云图
# ax.plot_surface(X, Y, u_values, cmap='viridis')
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('u')
# # 显示图形
# plt.show()

dx = dy = 0.005
dxdy = dx*dy

x = np.arange(-0.25+0.0*dx,0.25+1.0*dx,dx)
y = np.arange(-0.75+0.0*dy,-0.25+0.0*dy,dy)
X, Y = np.meshgrid(x, y)

X_flat = X.flatten()
Y_flat = Y.flatten()

x_area = np.column_stack((X_flat, Y_flat))
u_ic = -np.tanh(((x_area[:,0:1]**2+(x_area[:,1:2]+0.5)**2)**.5-0.15)/(2**.5*0.01))

neighborhoods, distances, distance_vectors,neighborhood_radius = find_neighborhood_points(x_area, k_neighbors=29)
h = neighborhood_radius*1.1/2.0

kernel = sph_kernel(distances, h)

RKPM_C2 = Compute_C(distance_vectors,kernel.unsqueeze(-1),dxdy,2)
f_d_RKPM2 = R_compute_field_gradients(torch.tensor(u_ic, dtype=torch.double),kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C2)
f_d_RKPM3 = R_compute_field_gradients(f_d_RKPM2[:,2:3],kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C2)

RKPM_C4 = Compute_C2(distance_vectors,kernel.unsqueeze(-1),dxdy,4)
f_d_RKPM4 = R_compute_field_gradients(torch.tensor(u_ic, dtype=torch.double),kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C4)

RKPM_C24 = Compute_C(distance_vectors,kernel.unsqueeze(-1),dxdy,4)
f_d_RKPM24 = R_compute_field_gradients(torch.tensor(u_ic, dtype=torch.double),kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C24)

u_values = (f_d_RKPM3[:,2:3])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area[:,0:1], x_area[:,1:2], c = u_values,s=5, cmap='jet')
plt.colorbar()
plt.show()
print(np.abs(u_values).max())

u_values = (f_d_RKPM4[:,0:1])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area[:,0:1], x_area[:,1:2], c = u_values,s=5, cmap='jet')
plt.colorbar()
plt.show()
print(np.abs(u_values).max())


u_values = (f_d_RKPM2[:,2:3])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area[:,0:1], x_area[:,1:2], c = u_values,s=5, cmap='jet')
plt.colorbar()
plt.show()
print(np.abs(u_values).max())
u_values = (f_d_RKPM24[:,2:3])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area[:,0:1], x_area[:,1:2], c = u_values,s=5, cmap='jet')
plt.colorbar()
plt.show()
print(np.abs(u_values).max())

u_values = (f_d_RKPM4[:,0:1]+2*f_d_RKPM4[:,1:2]+f_d_RKPM4[:,2:3])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area[:,0:1], x_area[:,1:2], c = u_values,s=5, cmap='jet')
plt.colorbar()
plt.show()
print(np.abs(u_values).max())