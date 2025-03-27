import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



mat = pd.read_excel('Re400.xlsx').to_numpy()
V_Ture = mat[:,2:3]
X_area = np.hstack((mat[:,0:1],mat[:,1:2]))
V_PAD = np.load('V_PAD.npy')
V_PmAD = np.load('V_PmAD.npy')
V_PSK_FT_Grad = np.load('V_PSK_FT_Grad.npy')
V_PSKandAD = np.load('V_PSKandAD.npy')

plt.figure()
plt.scatter(X_area[:,0], X_area[:,1], c = V_Ture[:,0],s=0.8, cmap='jet',vmax=1.0,vmin=0.0)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(X_area[:,0], X_area[:,1], c = np.abs(V_Ture[:,0]-V_PAD[:,0]),s=0.8, cmap='jet',vmax=0.05,vmin=0.0)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(X_area[:,0], X_area[:,1], c = np.abs(V_Ture[:,0]-V_PmAD[:,0]),s=0.8, cmap='jet',vmax=0.05,vmin=0.0)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(X_area[:,0], X_area[:,1], c = np.abs(V_Ture[:,0]-V_PSK_FT_Grad[:,0]),s=0.8, cmap='jet',vmax=0.05,vmin=0.0)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()

plt.figure()
plt.scatter(X_area[:,0], X_area[:,1], c = np.abs(V_Ture[:,0]-V_PSKandAD[:,0]),s=0.8, cmap='jet',vmax=0.05,vmin=0.0)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()

print("AD L2 error:", np.linalg.norm(V_Ture-V_PAD)/np.linalg.norm(V_Ture))
print("mAD L2 error:", np.linalg.norm(V_Ture-V_PmAD)/np.linalg.norm(V_Ture))
print("SK L2 error:", np.linalg.norm(V_Ture-V_PSK_FT_Grad)/np.linalg.norm(V_Ture))
print("SKandAD L2 error:", np.linalg.norm(V_Ture-V_PSKandAD)/np.linalg.norm(V_Ture))