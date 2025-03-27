import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from torch.nn.utils.rnn import pad_sequence

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

def sph_kernel_gradient(distances, h):
    q = distances / h
    result = torch.zeros_like(distances, dtype=torch.float64)
    within_range = (0 <= q) & (q <= 2)
    result[within_range] = (15 / (7 * np.pi * h ** 3)) * (
            (-2.0 * q[within_range] + 1.5 * q[within_range] ** 2) * (q[within_range] <= 1) +
            (-0.5 * (2 - q[within_range]) ** 2) * ((1 < q[within_range]) & (q[within_range] <= 2))
    )
    return result

def compute_field(f_values,kernel, neighborhoods, distances,dxdy):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(u_ngr * kernel * dxdy,dim = 1)
    return field_gradients

def compute_field_gradients(f_values,kernel_gradients, neighborhoods, distances, distance_vectors,dxdy):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',distance_vectors,- u_ngr * kernel_gradients * dxdy/(distances+1e-10)),dim = 1)
    return field_gradients

def CSPHcompute_field_gradients(f_values,kernel_gradients, neighborhoods, distances, distance_vectors,dxdy):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',distance_vectors,- (u_ngr-f_values) * kernel_gradients * dxdy/(distances+1e-10)),dim = 1)
    a = distance_vectors * distance_vectors
    CSPH_field_gradients = field_gradients/torch.sum(torch.einsum('ijk,ij->ijk',distance_vectors * distance_vectors,- kernel_gradients * dxdy/(distances+1e-10)),dim = 1)
    return CSPH_field_gradients

def KGC_Matrix_inverse(kernel_gradients,distance_vectors,distances,dxdy):
    Moment_Vector1 = torch.cat([distance_vectors[:, :, 0:1],
                               distance_vectors[:, :, 1:2],], dim=2)
    kernel_Vector = torch.einsum('ijk,ij->ijk', distance_vectors, -  kernel_gradients * dxdy / (distances + 1e-10))
    Matrix = torch.matmul(kernel_Vector.unsqueeze(3), Moment_Vector1.unsqueeze(2))
    Matrix_sum = torch.sum(Matrix, dim=1)
    Matrix_inverse = torch.inverse(Matrix_sum)
    return Matrix_inverse

def KGC_compute_field_gradients(f_values,kernel_Vector, neighborhoods, distances, distance_vectors,dxdy,Matrix_inverse):
    u_ngr = f_values[neighborhoods][:,:,0]
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',distance_vectors,- (u_ngr-f_values) * kernel_Vector * dxdy/(distances+1e-10)),dim = 1)
    FPM_field_gradients = torch.matmul(Matrix_inverse, field_gradients.unsqueeze(-1)).squeeze(-1)
    return FPM_field_gradients

# 有限核粒子方法
# 先需要根据distance_vectors（维度N,n,2）组一个向量列表（维度N,n,最后一个维度根据需求调整），然后做求逆操作
def FPM_Matrix_inverse(kernel,kernel_gradients,distance_vectors,distances,dxdy):
    Moment_Vector1 = torch.cat([torch.ones(distances.shape).unsqueeze(-1),
                               distance_vectors], dim=2)
    a = kernel.unsqueeze(-1)
    kernel_Vector = torch.cat([kernel.unsqueeze(-1) * dxdy,
                                torch.einsum('ijk,ij->ijk',distance_vectors, - kernel_gradients * dxdy/(distances+1e-10))], dim=2)
    Matrix = torch.matmul(kernel_Vector.unsqueeze(3), Moment_Vector1.unsqueeze(2))
    Matrix_sum = torch.sum(Matrix, dim=1)
    Matrix_inverse = torch.inverse(Matrix_sum)
    return Matrix_inverse,kernel_Vector

def FPM_compute_field_gradients(f_values,kernel_Vector, neighborhoods, distances, distance_vectors,dxdy,Matrix_inverse):
    u_ngr = f_values[neighborhoods][:,:,0]*(neighborhoods != -1)
    # 将邻域内点的梯度值按照核函数梯度进行加权求和
    field_gradients = torch.sum(torch.einsum('ijk,ij->ijk',kernel_Vector,(u_ngr-f_values)),dim = 1)
    FPM_field_gradients = torch.matmul(Matrix_inverse[:,1:3,:], field_gradients.unsqueeze(-1)).squeeze(-1)
    return FPM_field_gradients


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

## 主函数
dx = dy = 0.02
dxdy = dx*dy
ub = np.array([1,1])
lb = np.array([-1,-1])
x = np.arange(lb[0]-0.0*dx,ub[0]+1.0*dx,dx)
y = np.arange(lb[1]-0.0*dy,ub[1]+1.0*dy,dy)
X,Y = np.meshgrid(x,y)
x_area = np.hstack((X.reshape(-1,1),Y.reshape(-1,1)))
x_area_tensor = torch.from_numpy(x_area)

# num_points = 20000  # 你想要的点的数量
# ub = np.array([1,1])
# lb = np.array([-1,-1])
# dxdy = (ub[0]-lb[0])*(ub[1]-lb[1])/num_points
# # 生成随机坐标
# x = np.random.uniform(lb[0], ub[0], num_points)
# y = np.random.uniform(lb[1], ub[1], num_points)
# x_area = np.column_stack((x, y))
# # 将坐标转换为张量
# x_area_tensor = torch.tensor(x_area, dtype=torch.double)





u = f(x_area_tensor[:,0:1], x_area_tensor[:,1:2])
h = dxdy ** 0.5 * 2.1
neighborhoods, distances, distance_vectors = find_neighborhood_points(x_area, 2.0*h)
kernel = sph_kernel(distances, h)
kernel_gradients = sph_kernel_gradient(distances, h)
f_d = compute_field_gradients(u,kernel_gradients, neighborhoods, distances, distance_vectors,dxdy)
f_dxd = compute_field_gradients(f_d[:,0:1],kernel_gradients, neighborhoods, distances, distance_vectors,dxdy)

f_d_CSPH = CSPHcompute_field_gradients(u,kernel_gradients, neighborhoods, distances, distance_vectors,dxdy)
f_dxd_CSPH= CSPHcompute_field_gradients(f_d_CSPH[:,0:1],kernel_gradients, neighborhoods, distances, distance_vectors,dxdy)

KGC = KGC_Matrix_inverse(kernel_gradients,distance_vectors,distances,dxdy)
f_d_KGC = KGC_compute_field_gradients(u,kernel_gradients, neighborhoods, distances, distance_vectors,dxdy,KGC)
f_dxd_KGC= CSPHcompute_field_gradients(f_d_KGC[:,0:1],kernel_gradients, neighborhoods, distances, distance_vectors,dxdy)

FPM,kernel_Vector = FPM_Matrix_inverse(kernel,kernel_gradients,distance_vectors,distances,dxdy)
f_d_FPM = FPM_compute_field_gradients(u,kernel_Vector, neighborhoods, distances, distance_vectors,dxdy,FPM)
f_dxd_FPM = FPM_compute_field_gradients(f_d_FPM[:,0:1],kernel_Vector, neighborhoods, distances, distance_vectors,dxdy,FPM)


RKPM_C2 = Compute_C(distance_vectors,kernel.unsqueeze(-1),dxdy,2)
f_d_RKPM2 = R_compute_field_gradients(u,kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C2)

RKPM_C3 = Compute_C(distance_vectors,kernel.unsqueeze(-1),dxdy,3)
f_d_RKPM3 = R_compute_field_gradients(u,kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C3)

RKPM_C4 = Compute_C(distance_vectors,kernel.unsqueeze(-1),dxdy,4)
f_d_RKPM4 = R_compute_field_gradients(u,kernel.unsqueeze(-1), neighborhoods, distances, distance_vectors,dxdy,RKPM_C4)



fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2]),s=5, cmap='jet')
plt.colorbar()
plt.show()

u_values = (f_dxd[:,1:2])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
# - f_dx(x_area_tensor[:,0:1], x_area_tensor[:,1:2])
u_values = (f_dxd_CSPH[:,1:2])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
u_values = (f_dxd_KGC[:,1:2])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
u_values = (f_dxd_FPM[:,1:2])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
u_values = (f_d_RKPM2[:,3:4])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
u_values = (f_d_RKPM3[:,3:4])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())
u_values = (f_d_RKPM4[:,3:4])
fig = plt.figure()
# ax.plot_surface(X, Y, u_values, cmap='viridis',vmin=-1,vmax=1)
plt.scatter(x_area_tensor[:,0:1], x_area_tensor[:,1:2], c = np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])),s=5, cmap='jet',vmax=50.0,vmin=0.0)
plt.colorbar()
plt.show()
print(np.abs(u_values-f_dxy(x_area_tensor[:,0:1], x_area_tensor[:,1:2])).max())

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
