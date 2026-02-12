import os

# 确保正确设置 PyTorch 作为后端
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from deepxde.backend import torch
from deepxde.nn import activations
import re

# ============================================================================
# 1. 配置与网络定义 (Configuration & Network)
# ============================================================================

# 如果有可用的 GPU，则指定使用第一个 GPU
if torch.cuda.is_available():
    torch.cuda.set_device(0)


class MPFNN(dde.nn.PFNN):
    def __init__(self, layer_sizes, second_layer_sizes, activation, kernel_initializer):
        super(MPFNN, self).__init__(layer_sizes, activation, kernel_initializer)
        self.first_layer_sizes = layer_sizes
        self.second_layer_sizes = second_layer_sizes
        self.activation = activations.get(activation)
        self.firstFNN = dde.nn.PFNN(self.first_layer_sizes, self.activation, kernel_initializer)
        self.secondFNN = dde.nn.PFNN(self.second_layer_sizes, self.activation, kernel_initializer)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        x_firstFNN = self.firstFNN(x)
        x_secondFNN = self.secondFNN(x)
        x = torch.cat((x_firstFNN, x_secondFNN), dim=1)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


# ============================================================================
# 2. 数据加载与处理 (Data Loading)
# ============================================================================

# 检查数据文件
data_filename = "/code/zsn/Code3/Code1/sim_data/hollow_cylinder_data.npy"
if not os.path.exists(data_filename):
    # 尝试在上一级目录查找
    if os.path.exists("../data/" + data_filename):
        data_filename = "../data/" + data_filename
    else:
        raise FileNotFoundError(f"找不到数据文件 {data_filename}，请先运行数据生成脚本！")

print(f"Loading data from: {data_filename}")
data_dict = np.load(data_filename, allow_pickle=True).item()

coors = data_dict["coordinates"]
gt_disp = data_dict["displacements"]
bnd_coors = data_dict["boundary_coordinates"]
bond_gt_disp = data_dict["boundary_displacements"]
radius = data_dict["radius"]

# 缩放位移
scale = 1e4
gt_disp = gt_disp * scale
bond_gt_disp = bond_gt_disp * scale

# 统计均值方差用于反归一化
ux_mean, uy_mean = np.mean(gt_disp[:, 0]), np.mean(gt_disp[:, 1])
ux_std, uy_std = np.std(gt_disp[:, 0]), np.std(gt_disp[:, 1])

# 采样训练点 (Sampling)
idx1 = np.random.choice(np.where(radius < 2)[0], 500, replace=False)
idx2 = np.random.choice(np.where((radius > 2) & (radius < 3))[0], 400, replace=False)
idx3 = np.random.choice(np.where((radius > 3) & (radius < 4))[0], 300, replace=False)
idx4 = np.random.choice(np.where((radius > 4) & (radius <= 5))[0], 200, replace=False)
idx5 = np.random.choice(np.where(bnd_coors)[0], 150, replace=False)

pde_pts = np.vstack((coors[idx1, :], coors[idx2, :], coors[idx3, :], coors[idx4, :], bnd_coors[idx5, :]))
pde_pts_disp = np.vstack(
    (gt_disp[idx1, :], gt_disp[idx2, :], gt_disp[idx3, :], gt_disp[idx4, :], bond_gt_disp[idx5, :]))

geom = dde.geometry.PointCloud(points=pde_pts)

# ============================================================================
# 3. 物理模型与训练配置 (Physics & Training)
# ============================================================================

losses = [dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1])]

# --- 这里的参数必须与生成数据时一致 ---
p_in = 1.5e-5 * scale
E_true = 0.15
nu_true = 0.3

E_ = dde.Variable(1.0)
nu_ = dde.Variable(1.0)


def strain(x, y):
    ux, uy = y[:, 0:1], y[:, 1:2]
    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    exy = 0.5 * (dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0))
    return exx, eyy, exy


def stress(x, y):
    exx, eyy, exy = strain(x, y)
    E = (torch.tanh(E_) + 1.0) / 2
    nu = (torch.tanh(nu_) + 1.0) / 4
    sxx = E / (1 - nu ** 2) * (exx + nu * eyy)
    syy = E / (1 - nu ** 2) * (eyy + nu * exx)
    sxy = E / (1 + nu) * exy
    return sxx, syy, sxy, E


def pde(x, y):
    Nsxx, Nsyy, Nsxy, Nsrr = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]
    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    sxy_y = dde.grad.jacobian(Nsxy, x, i=0, j=1)
    sxy_x = dde.grad.jacobian(Nsxy, x, i=0, j=0)
    mx = sxx_x + sxy_y
    my = sxy_x + syy_y
    sxx, syy, sxy, E = stress(x, y)
    nx = torch.cos(torch.arctan(x[:, 1:2] / x[:, 0:1]))
    ny = torch.sin(torch.arctan(x[:, 1:2] / x[:, 0:1]))
    srr = sxx * torch.square(nx) + syy * torch.square(ny) + sxy * 2 * nx * ny
    return mx, my, sxx - Nsxx, syy - Nsyy, sxy - Nsxy, srr - Nsrr


data = dde.data.PDE(geom, pde, losses, anchors=pde_pts)


def output_transform(x, y):
    Nux, Nuy = y[:, 0:1], y[:, 1:2]
    Nsxx, Nsyy, Nsxy, Nsrr = y[:, 2:3], y[:, 3:4], y[:, 4:5], y[:, 5:6]
    Nux = x[:, 0:1] * (Nux * ux_std + ux_mean)
    Nuy = x[:, 1:2] * (Nuy * uy_std + uy_mean)
    rad = torch.sqrt(torch.square(x[:, 0:1]) + torch.square(x[:, 1:2]))
    Nsxx = (rad - 5) * Nsxx
    Nsyy = (rad - 5) * Nsyy
    Nsxy = (rad - 5) * x[:, 0:1] * x[:, 1:2] * Nsxy
    Nsrr = ((rad - 1) * Nsrr - 1) * p_in / -4 * (rad - 5)
    return torch.concat([Nux, Nuy, Nsxx, Nsyy, Nsxy, Nsrr], axis=1)


net = MPFNN([2] + [45] * 5 + [2], [2] + [45] * 5 + [4], "swish", "Glorot normal")
net.apply_output_transform(output_transform)
model = dde.Model(data, net)

external_trainable_variables = [E_, nu_]
variables = dde.callbacks.VariableValue(external_trainable_variables, period=1000, filename="variables.dat")

model.compile("adam", lr=1e-3, decay=["step", 15000, 0.15], loss_weights=[1] * 2 + [1e1] * 4 + [1],
              external_trainable_variables=external_trainable_variables)

# 训练 (适当减少 epochs 以便演示，实际使用建议 50000+)
print("开始训练...")
losshistory, train_state = model.train(epochs=100000, callbacks=[variables])
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# ============================================================================
# 4. 结果可视化 (Visualization B, C, D)
# ============================================================================
print("开始生成可视化图片...")

# --- Figure B: Training Loss ---
plt.figure(figsize=(6, 4))
# DeepXDE loss_train 是一个 shape (epochs, num_losses) 的数组，求和得到 Total Loss
loss_train = np.sum(losshistory.loss_train, axis=1)
plt.semilogy(loss_train, 'b-', label='Total Loss')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Training loss', fontsize=12)
plt.title('B: Training Loss History')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.tight_layout()
plt.savefig("Fig_B_Loss.png", dpi=300)
print("Saved: Fig_B_Loss.png")

# --- Figure C: Parameter Estimation ---
if os.path.exists("variables.dat"):
    lines = open("variables.dat", "r").readlines()
    # 解析 variables.dat
    records = []
    steps = []
    for i, line in enumerate(lines):
        # 提取步数 (DeepXDE 格式通常第一列是 step, 或者是按照 period 记录)
        # VariableValue callback 记录格式: [step, [val1, val2]]
        # 但写入文件通常是: step [v1, v2] 或 [v1, v2] (取决于版本)
        # 这里我们假设按行数对应 period=1000
        steps.append(i * 1000)
        vals_str = min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len)
        vals = np.fromstring(vals_str, sep=",")
        records.append(vals)

    records = np.array(records)

    # 还原物理参数
    E_hist = (np.tanh(records[:, 0]) + 1.0) / 2
    nu_hist = (np.tanh(records[:, 1]) + 1.0) / 4

    plt.figure(figsize=(6, 4))
    plt.plot(steps, E_hist, 'b-', linewidth=2, label='E (Predicted)')
    plt.plot(steps, nu_hist, 'g-', linewidth=2, label=r'$\nu$ (Predicted)')

    # 绘制真实值参考线
    plt.axhline(y=E_true, color='b', linestyle='--', alpha=0.5, label='E (True)')
    plt.axhline(y=nu_true, color='g', linestyle='--', alpha=0.5, label=r'$\nu$ (True)')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(r'E ($N/\mu m^2$) / $\nu$', fontsize=12)
    plt.title('C: Parameter Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Fig_C_Parameters.png", dpi=300)
    print("Saved: Fig_C_Parameters.png")

    # 打印最终误差
    print(f"Final E: {E_hist[-1]:.4f} (True: {E_true}), Error: {abs(E_hist[-1] - E_true) / E_true * 100:.2f}%")
    print(f"Final nu: {nu_hist[-1]:.4f} (True: {nu_true}), Error: {abs(nu_hist[-1] - nu_true) / nu_true * 100:.2f}%")

# --- Figure D: PINN Verification & Error (后两列) ---
# 使用整个数据集的坐标进行预测
X_all = coors
# 预测
y_pred_all = model.predict(X_all)
# 提取预测位移 (前两个通道)
# 注意：model.predict 已经自动调用了 output_transform，所以已经是物理值
ux_pred = y_pred_all[:, 0]
uy_pred = y_pred_all[:, 1]

# 真实位移 (已经由数据加载部分 scale 过)
# gt_disp 是 shape (N, 2)
ux_true = gt_disp[:, 0]
uy_true = gt_disp[:, 1]

# 计算误差
err_ux = np.abs(ux_true - ux_pred)
err_uy = np.abs(uy_true - uy_pred)


# 绘图函数
def plot_field(x, y, c, title, filename, unit, cmap='jet'):
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(x, y, c=c, cmap=cmap, s=2)
    plt.colorbar(sc, label=unit)
    plt.axis('equal')
    plt.xlabel(r'x ($\mu m$)')
    plt.ylabel(r'y ($\mu m$)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


# 1. PINN Prediction (图中中间一列)
plot_field(coors[:, 0], coors[:, 1], ux_pred, r'PINN Prediction: $u_x$', 'Fig_D_PINN_ux.png', r'$\mu m$')
plot_field(coors[:, 0], coors[:, 1], uy_pred, r'PINN Prediction: $u_y$', 'Fig_D_PINN_uy.png', r'$\mu m$')

# 2. Pointwise Error (图中最后一列)
plot_field(coors[:, 0], coors[:, 1], err_ux, r'Pointwise Error: $u_x$', 'Fig_D_Error_ux.png', r'$\mu m$')
plot_field(coors[:, 0], coors[:, 1], err_uy, r'Pointwise Error: $u_y$', 'Fig_D_Error_uy.png', r'$\mu m$')

print("所有图片已保存完毕。")