import os

# ==============================================================================
# 0. 环境配置
# ==============================================================================
os.environ["DDEBACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch
from deepxde.nn import activations
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 如果有 GPU 则使用 (根据你的环境选择 ID，通常是 0)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

dde.config.set_random_seed(603)

# ==============================================================================
# 1. 读取仿真数据 & 预设参数
# ==============================================================================
# 确保文件名与仿真代码保存的一致
# 优先使用绝对路径，如果不存在则尝试当前目录
default_path = "/code/zsn/Code3/Code1/sim_data/annulus_pinn_data_newV.npy"
if os.path.exists(default_path):
    data_path = default_path
else:
    data_path = "annulus_pinn_data_newV.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件，请检查路径: {default_path}")

print(f"正在加载数据: {data_path}")
data = np.load(data_path, allow_pickle=True).item()

coors = data["coordinates"]  # 所有节点
gt_disp = data["displacements"]
mesh_elements = data["elements"]

# 提取边界点 (BC用)
bnd_indices = data["boundary_indices"]
bnd_coors = coors[bnd_indices]
bnd_gt_disp = gt_disp[bnd_indices]

# 圆环几何参数（与仿真代码严格一致）
R_INNER = 3.0
R_OUTER = 5.0
PRESSURE_MAG = 5.0e-3
s_scale = 1e-3
ux_std = np.std(gt_disp[:, 0])
uy_std = np.std(gt_disp[:, 1])

# 准备真值数据 (用于后续对比绘图)
true_sxx = data["stress_xx"].flatten()
true_syy = data["stress_yy"].flatten()
true_sxy = data["stress_xy"].flatten()

print(f"✓ 数据加载完成: {len(coors)} Nodes, {len(mesh_elements)} Elements")

# ==============================================================================
# 1.5 预计算：计算三角形的重心 (PDE 训练点) & 区域掩码
# ==============================================================================
num_elements = len(mesh_elements)
element_centers = np.zeros((num_elements, 2))
for i, el in enumerate(mesh_elements):
    element_centers[i] = np.mean(coors[el], axis=0)

print(f"✓ 单元中心计算完成，共 {num_elements} 个点 (用作 PDE 训练点)")

# --- 提前生成区域 Mask (逻辑与仿真代码完全一致) ---
# 注意：这些 Mask 仅用于后续的"画图对比"和"统计误差"，
# 网络训练本身是完全不知道这些区域划分的 (Blind Inversion)。
base_mask = []
red_mask = []
blue_mask = []

for i, center in enumerate(element_centers):
    cx, cy = center
    r = np.sqrt(cx ** 2 + cy ** 2)
    theta = np.arctan2(cy, cx)

    # 红色区域逻辑 (仿真: 2.6 <= theta <= 3.7 and 3.5 <= r <= 4.5)
    if (2.6 <= theta <= 3.7) and (3.5 <= r <= 4.5):
        red_mask.append(i)
    # 蓝色区域逻辑 (仿真: -1.0 <= theta <= -0.35 and 3.5 <= r <= 4.5)
    elif (-1.0 <= theta <= -0.35) and (3.5 <= r <= 4.5):
        blue_mask.append(i)
    else:
        base_mask.append(i)

# 转为 array 方便索引
base_mask = np.array(base_mask)
red_mask = np.array(red_mask)
blue_mask = np.array(blue_mask)

print(f"✓ 区域划分(仅用于验证): Base={len(base_mask)}, Red={len(red_mask)}, Blue={len(blue_mask)}")


# ==============================================================================
# 2. 网络结构
# ==============================================================================
class MPFNN(dde.nn.PFNN):
    def __init__(self, layer_sizes, second_layer_sizes, activation, kernel_initializer):
        super(MPFNN, self).__init__(layer_sizes, activation, kernel_initializer)
        self.firstFNN = dde.nn.PFNN(layer_sizes, activations.get(activation), kernel_initializer)
        self.secondFNN = dde.nn.PFNN(second_layer_sizes, activations.get(activation), kernel_initializer)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        x_disp = self.firstFNN(x)
        x_stress = self.secondFNN(x)
        x_out = torch.cat((x_disp, x_stress), dim=1)
        if self._output_transform is not None:
            x_out = self._output_transform(inputs, x_out)
        return x_out


# ==============================================================================
# 3. 输出变换（强制硬约束）
# ==============================================================================
def output_transform(x, y):
    x_loc, y_loc = x[:, 0:1], x[:, 1:2]
    r = torch.sqrt(x_loc ** 2 + y_loc ** 2)

    # 仿真边界条件: 外边界固定 (u=0 at R_OUTER)
    # 权重函数 w_disp: 在 R_OUTER 处为 0，在 R_INNER 处为 1
    w_disp = (R_OUTER - r) / (R_OUTER - R_INNER)

    u_final = y[:, 0:1] * ux_std * w_disp
    v_final = y[:, 1:2] * uy_std * w_disp
    sxx_final = y[:, 2:3] * s_scale
    syy_final = y[:, 3:4] * s_scale
    txy_final = y[:, 4:5] * s_scale
    return torch.cat([u_final, v_final, sxx_final, syy_final, txy_final], dim=1)


# ==============================================================================
# 4. 向量化参数查找 (极速版 & 盲反演)
# ==============================================================================
element_centers_tensor = torch.tensor(element_centers, dtype=torch.float32)
if torch.cuda.is_available():
    element_centers_tensor = element_centers_tensor.cuda()

# 初始化为全零，网络需要自己学习出每个单元的值
# num_elements 为单元总数，每个单元有独立的 E 和 nu
_E_tensor = torch.zeros((num_elements, 1), dtype=torch.float32)
_nu_tensor = torch.zeros((num_elements, 1), dtype=torch.float32)

if torch.cuda.is_available():
    _E_tensor = _E_tensor.cuda()
    _nu_tensor = _nu_tensor.cuda()

E_vars = dde.Variable(_E_tensor)
nu_vars = dde.Variable(_nu_tensor)


def stress(x, y):
    # 1. 计算应变
    ux, uy = y[:, 0:1], y[:, 1:2]
    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    gxy = dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)

    # 2. 查找索引 (找到当前坐标 x 属于哪个单元)
    dist = torch.cdist(x, element_centers_tensor)
    indices = torch.argmin(dist, dim=1)

    # 3. 参数映射 (Tanh)
    def map_param(var_tensor, min_val, max_val):
        return min_val + (torch.tanh(var_tensor) + 1.0) / 2.0 * (max_val - min_val)

    # 4. 取值
    batch_E_raw = E_vars[indices]
    batch_nu_raw = nu_vars[indices]

    # 批量映射 (范围: E[0.2, 1.3], Nu[0.1, 0.6])
    E_field = map_param(batch_E_raw, 0.2, 1.3)
    nu_field = map_param(batch_nu_raw, 0.1, 0.6)

    # 5. 本构方程 (平面应变 Plane Strain)
    factor = E_field / ((1 + nu_field) * (1 - 2 * nu_field))
    sxx_phys = factor * ((1 - nu_field) * exx + nu_field * eyy)
    syy_phys = factor * (nu_field * exx + (1 - nu_field) * eyy)
    txy_phys = factor * ((1 - 2 * nu_field) / 2.0 * gxy)

    return sxx_phys, syy_phys, txy_phys, E_field


# ==============================================================================
# 5. PDE Loss
# ==============================================================================
def pde(x, y):
    Nsxx, Nsyy, Ntxy = y[:, 2:3], y[:, 3:4], y[:, 4:5]
    sxx_x = dde.grad.jacobian(Nsxx, x, i=0, j=0)
    syy_y = dde.grad.jacobian(Nsyy, x, i=0, j=1)
    txy_x = dde.grad.jacobian(Ntxy, x, i=0, j=0)
    txy_y = dde.grad.jacobian(Ntxy, x, i=0, j=1)

    momentum_x = (sxx_x + txy_y)
    momentum_y = (txy_x + syy_y)

    sxx_phys, syy_phys, txy_phys, _ = stress(x, y)

    stress_xx_res = (sxx_phys - Nsxx)
    stress_yy_res = (syy_phys - Nsyy)
    stress_xy_res = (txy_phys - Ntxy)

    return [momentum_x, momentum_y, stress_xx_res, stress_yy_res, stress_xy_res]


# ==============================================================================
# 6. 数据配置 (Split Strategy)
# ==============================================================================
pde_pts = element_centers
geom = dde.geometry.PointCloud(points=pde_pts)
bc_disp = dde.PointSetBC(coors, gt_disp, component=[0, 1])

data_obj = dde.data.PDE(
    geom,
    pde,
    [bc_disp],
    anchors=pde_pts
)

net = MPFNN([2] + [50] * 5 + [2], [2] + [50] * 5 + [3], "swish", "Glorot normal")
net.apply_output_transform(output_transform)
model = dde.Model(data_obj, net)


# ==============================================================================
# 7. 训练配置 (含历史记录功能)
# ==============================================================================

# 参数映射函数 (Numpy版，用于可视化和记录，需与 forward 中保持一致)
def map_val_E_np(v):
    return 0.2 + (np.tanh(v) + 1.0) / 2.0 * 1.1  # Range: [0.2, 1.3]

def map_val_nu_np(v):
    return 0.1 + (np.tanh(v) + 1.0) / 2.0 * 0.5  # Range: [0.1, 0.6]

class HistoryRecorder(dde.callbacks.Callback):
    def __init__(self, filename, period=1000):
        super(HistoryRecorder, self).__init__()
        self.filename = filename
        self.period = period
        self.history = {
            'steps': [],
            'E_base': [], 'E_red': [], 'E_blue': [],
            'nu_base': [], 'nu_red': [], 'nu_blue': []
        }

    def on_epoch_end(self):
        epoch = self.model.train_state.epoch
        if epoch % self.period == 0:
            e_raw = E_vars.detach().cpu().numpy().flatten()
            nu_raw = nu_vars.detach().cpu().numpy().flatten()
            e_phys = map_val_E_np(e_raw)
            nu_phys = map_val_nu_np(nu_raw)

            self.history['steps'].append(epoch)
            self.history['E_base'].append(np.mean(e_phys[base_mask]))
            self.history['E_red'].append(np.mean(e_phys[red_mask]))
            self.history['E_blue'].append(np.mean(e_phys[blue_mask]))

            self.history['nu_base'].append(np.mean(nu_phys[base_mask]))
            self.history['nu_red'].append(np.mean(nu_phys[red_mask]))
            self.history['nu_blue'].append(np.mean(nu_phys[blue_mask]))

            np.savez(self.filename, E=e_raw, nu=nu_raw, history=self.history)


fname_var = "element_params_optimized.npz"
recorder = HistoryRecorder(fname_var, period=1000)

# ==============================================================================
# [关键更新] 真实材料参数 (用于验证和画图)
# ==============================================================================
TRUE_PARAMS = {
    "Base": (0.7, 0.34),   # 仿真: E=0.7, NU=0.34
    "Red":  (0.9, 0.45),   # 仿真修改: E=0.9, NU=0.45
    "Blue": (0.45, 0.30)   # 仿真: E=0.45, NU=0.3
}

print(f"\n开始训练 (Split Strategy + 向量化极速版)...")

model.compile(
    "adam",
    lr=1e-3,
    decay=["step", 15000, 0.5],
    loss_weights=[1, 1, 1, 1, 1, 20],
    external_trainable_variables=[E_vars, nu_vars]
)

losshistory, train_state = model.train(
    iterations=100000,
    callbacks=[recorder],
    display_every=1000
)
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# ==============================================================================
# 8. 结果可视化：参数收敛曲线
# ==============================================================================
print("\n" + "=" * 80)
print("1. 生成参数收敛曲线...")
print("=" * 80)

hist = recorder.history
steps = hist['steps']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制 E 收敛
ax1.plot(steps, hist['E_base'], 'k-', label='Base Pred')
ax1.plot(steps, hist['E_red'], 'r-', label='Red Pred')
ax1.plot(steps, hist['E_blue'], 'b-', label='Blue Pred')
ax1.axhline(TRUE_PARAMS['Base'][0], color='k', linestyle='--', alpha=0.5, label='Base True')
ax1.axhline(TRUE_PARAMS['Red'][0], color='r', linestyle='--', alpha=0.5, label='Red True')
ax1.axhline(TRUE_PARAMS['Blue'][0], color='b', linestyle='--', alpha=0.5, label='Blue True')
ax1.set_xlabel('Iteration')
ax1.set_ylabel("Young's Modulus E (Pa)")
ax1.set_title("Convergence of E")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 绘制 Nu 收敛
ax2.plot(steps, hist['nu_base'], 'k-', label='Base Pred')
ax2.plot(steps, hist['nu_red'], 'r-', label='Red Pred')
ax2.plot(steps, hist['nu_blue'], 'b-', label='Blue Pred')
ax2.axhline(TRUE_PARAMS['Base'][1], color='k', linestyle='--', alpha=0.5)
ax2.axhline(TRUE_PARAMS['Red'][1], color='r', linestyle='--', alpha=0.5)
ax2.axhline(TRUE_PARAMS['Blue'][1], color='b', linestyle='--', alpha=0.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel("Poisson's Ratio Nu")
ax2.set_title("Convergence of Nu")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Fig_Param_Convergence.png", dpi=300)
print("   -> 已保存: Fig_Param_Convergence.png")

# ==============================================================================
# 9. 结果可视化：物理场对比 (含材料参数真值对比) - 统一色卡范围
# ==============================================================================
print("\n" + "=" * 80)
print("2. 生成物理场对比图 (Original vs Deformed & Parameters)...")
print("=" * 80)

# --------------------------------------------------------------------------
# A. 准备预测数据
# --------------------------------------------------------------------------
if os.path.exists(fname_var):
    loaded = np.load(fname_var, allow_pickle=True)
    final_E_raw = loaded["E"]
    final_nu_raw = loaded["nu"]
else:
    final_E_raw = E_vars.detach().cpu().numpy().flatten()
    final_nu_raw = nu_vars.detach().cpu().numpy().flatten()

# 映射回物理空间
element_E_pred = map_val_E_np(final_E_raw)
element_nu_pred = map_val_nu_np(final_nu_raw)

# 全场物理场预测
y_pred_mesh = model.predict(coors)
pred_ux = y_pred_mesh[:, 0]
pred_uy = y_pred_mesh[:, 1]
pred_sxx = y_pred_mesh[:, 2]
pred_syy = y_pred_mesh[:, 3]
pred_sxy = y_pred_mesh[:, 4]

# --------------------------------------------------------------------------
# B. 准备真值数据 (材料参数) - 用于对比
# --------------------------------------------------------------------------
# 构建全场真值数组
element_E_true = np.zeros(num_elements)
element_nu_true = np.zeros(num_elements)

# 填入 Base 值
element_E_true[base_mask] = TRUE_PARAMS["Base"][0]
element_nu_true[base_mask] = TRUE_PARAMS["Base"][1]

# 填入 Red 值
element_E_true[red_mask] = TRUE_PARAMS["Red"][0]
element_nu_true[red_mask] = TRUE_PARAMS["Red"][1]

# 填入 Blue 值
element_E_true[blue_mask] = TRUE_PARAMS["Blue"][0]
element_nu_true[blue_mask] = TRUE_PARAMS["Blue"][1]

print("   -> 已构建材料参数真值场 (True Parameter Fields constructed).")

# --------------------------------------------------------------------------
# C. 准备绘图网格
# --------------------------------------------------------------------------
triang_orig = mtri.Triangulation(coors[:, 0], coors[:, 1], triangles=mesh_elements)

SCALE_FACTOR = 50.0
x_def_pred = coors[:, 0] + pred_ux * SCALE_FACTOR
y_def_pred = coors[:, 1] + pred_uy * SCALE_FACTOR
triang_def_pred = mtri.Triangulation(x_def_pred, y_def_pred, triangles=mesh_elements)

x_def_true = coors[:, 0] + gt_disp[:, 0] * SCALE_FACTOR
y_def_true = coors[:, 1] + gt_disp[:, 1] * SCALE_FACTOR
triang_def_true = mtri.Triangulation(x_def_true, y_def_true, triangles=mesh_elements)

# --------------------------------------------------------------------------
# D. 计算全局统一色卡范围 (Unified Color Limits)
# --------------------------------------------------------------------------
# 1. 位移场
disp_vmin = min(np.min(gt_disp[:, 0]), np.min(gt_disp[:, 1]), np.min(pred_ux), np.min(pred_uy))
disp_vmax = max(np.max(gt_disp[:, 0]), np.max(gt_disp[:, 1]), np.max(pred_ux), np.max(pred_uy))

# 2. 应力场
stress_vmin = min(np.min(true_sxx), np.min(true_syy), np.min(true_sxy),
                  np.min(pred_sxx), np.min(pred_syy), np.min(pred_sxy))
stress_vmax = max(np.max(true_sxx), np.max(true_syy), np.max(true_sxy),
                  np.max(pred_sxx), np.max(pred_syy), np.max(pred_sxy))

# 3. 材料参数 (关键：取真值和预测值的并集范围)
E_vmin = min(np.min(element_E_pred), np.min(element_E_true))
E_vmax = max(np.max(element_E_pred), np.max(element_E_true))

nu_vmin = min(np.min(element_nu_pred), np.min(element_nu_true))
nu_vmax = max(np.max(element_nu_pred), np.max(element_nu_true))

print(f"   统一色卡范围:")
print(f"   - 位移场: [{disp_vmin:.6e}, {disp_vmax:.6e}]")
print(f"   - 应力场: [{stress_vmin:.6e}, {stress_vmax:.6e}]")
print(f"   - E场 (Pred & True): [{E_vmin:.4f}, {E_vmax:.4f}]")
print(f"   - Nu场 (Pred & True): [{nu_vmin:.4f}, {nu_vmax:.4f}]")

# --------------------------------------------------------------------------
# E. 绘图函数
# --------------------------------------------------------------------------
def save_single_field(triang_obj, values, title, filename, unit, vmin=None, vmax=None, cmap='jet', show_mesh=True):
    plt.figure(figsize=(7, 6), facecolor='white')
    if vmin is None or vmax is None or vmin == vmax:
        levels = 21
    else:
        levels = np.linspace(vmin, vmax, 21)

    cnt = plt.tricontourf(triang_obj, values, levels=levels, cmap=cmap, extend='both')
    if show_mesh:
        plt.triplot(triang_obj, 'k-', linewidth=0.3, alpha=0.5)
    cbar = plt.colorbar(cnt, fraction=0.046, pad=0.04)
    cbar.set_label(unit, fontsize=12)
    cbar.formatter.set_powerlimits((-2, 2))
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> 已保存: {filename}")

def save_all_combinations(val_true, val_pred, name_short, name_full, unit, vmin, vmax):
    # 原始网格
    save_single_field(triang_orig, val_true, f"GT (Orig): {name_full}", f"Orig_GT_{name_short}.png", unit, vmin=vmin, vmax=vmax)
    save_single_field(triang_orig, val_pred, f"Pred (Orig): {name_full}", f"Orig_Pred_{name_short}.png", unit, vmin=vmin, vmax=vmax)
    # 变形网格
    save_single_field(triang_def_true, val_true, f"GT (Deformed): {name_full}", f"Def_GT_{name_short}.png", unit, vmin=vmin, vmax=vmax)
    save_single_field(triang_def_pred, val_pred, f"Pred (Deformed): {name_full}", f"Def_Pred_{name_short}.png", unit, vmin=vmin, vmax=vmax)

# 生成位移图
save_all_combinations(gt_disp[:, 0], pred_ux, "Ux", "Disp X", "m", disp_vmin, disp_vmax)
save_all_combinations(gt_disp[:, 1], pred_uy, "Uy", "Disp Y", "m", disp_vmin, disp_vmax)

# 生成应力图
save_all_combinations(true_sxx, pred_sxx, "Sxx", "Stress XX", "Pa", stress_vmin, stress_vmax)
save_all_combinations(true_syy, pred_syy, "Syy", "Stress YY", "Pa", stress_vmin, stress_vmax)
save_all_combinations(true_sxy, pred_sxy, "Sxy", "Stress XY", "Pa", stress_vmin, stress_vmax)

# --------------------------------------------------------------------------
# F. 绘制材料参数 (True vs Pred) - 使用统一色卡
# --------------------------------------------------------------------------
def save_element_plot(triang_obj, values, title, filename, unit, vmin, vmax):
    plt.figure(figsize=(7, 6), facecolor='white')
    tpc = plt.tripcolor(triang_obj, facecolors=values, cmap='jet', edgecolors='k', linewidth=0.2, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(tpc, fraction=0.046, pad=0.04)
    cbar.set_label(unit, fontsize=12)
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> 已保存: {filename}")

# 绘制 E (预测 vs 真值)
save_element_plot(triang_orig, element_E_pred, "Predicted E Field", "Orig_Pred_E_Field.png", "E (Pa)", E_vmin, E_vmax)
save_element_plot(triang_orig, element_E_true, "True E Field", "Orig_True_E_Field.png", "E (Pa)", E_vmin, E_vmax)

# 绘制 Nu (预测 vs 真值)
save_element_plot(triang_orig, element_nu_pred, "Predicted Nu Field", "Orig_Pred_Nu_Field.png", "Nu", nu_vmin, nu_vmax)
save_element_plot(triang_orig, element_nu_true, "True Nu Field", "Orig_True_Nu_Field.png", "Nu", nu_vmin, nu_vmax)


# ==============================================================================
# 10. 结果可视化：逐点误差分析
# ==============================================================================
print("\n" + "=" * 80)
print("3. 生成逐点误差云图 (Pointwise Error)...")
print("=" * 80)

# 计算绝对误差
err_ux = np.abs(pred_ux - gt_disp[:, 0])
err_uy = np.abs(pred_uy - gt_disp[:, 1])
err_sxx = np.abs(pred_sxx - true_sxx)
err_syy = np.abs(pred_syy - true_syy)
err_sxy = np.abs(pred_sxy - true_sxy)

# 统一误差色卡
disp_err_vmax = max(np.max(err_ux), np.max(err_uy))
stress_err_vmax = max(np.max(err_sxx), np.max(err_syy), np.max(err_sxy))

print(f"   统一误差色卡范围:")
print(f"   - 位移误差: [0, {disp_err_vmax:.6e}]")
print(f"   - 应力误差: [0, {stress_err_vmax:.6e}]")

def plot_error_field(triang_obj, error_data, title, filename, unit, vmax, cmap='jet'):
    plt.figure(figsize=(7, 6), facecolor='white')
    levels = np.linspace(0, vmax, 21)
    cnt = plt.tricontourf(triang_obj, error_data, levels=levels, cmap=cmap, extend='max')
    plt.triplot(triang_obj, 'k-', linewidth=0.2, alpha=0.3)
    cbar = plt.colorbar(cnt, fraction=0.046, pad=0.04)
    cbar.set_label(f"Abs Error ({unit})", fontsize=12)
    cbar.formatter.set_powerlimits((-2, 2))
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> 已保存误差图: {filename}")

plot_error_field(triang_orig, err_ux, "Pointwise Error: Ux", "Fig_Error_Ux.png", "m", disp_err_vmax)
plot_error_field(triang_orig, err_uy, "Pointwise Error: Uy", "Fig_Error_Uy.png", "m", disp_err_vmax)
plot_error_field(triang_orig, err_sxx, "Pointwise Error: Sxx", "Fig_Error_Sxx.png", "Pa", stress_err_vmax)
plot_error_field(triang_orig, err_syy, "Pointwise Error: Syy", "Fig_Error_Syy.png", "Pa", stress_err_vmax)
plot_error_field(triang_orig, err_sxy, "Pointwise Error: Sxy", "Fig_Error_Sxy.png", "Pa", stress_err_vmax)

# ==============================================================================
# 11. 定量误差统计表
# ==============================================================================
print("\n" + "=" * 80)
print("4. 最终定量误差表...")
print("=" * 80)

def get_stats(mask, pred_e, pred_nu):
    if len(mask) == 0: return 0, 0, 0, 0
    return np.mean(pred_e[mask]), np.std(pred_e[mask]), \
           np.mean(pred_nu[mask]), np.std(pred_nu[mask])

stats_dict = {}
print(f"{'Region':<8} | {'Count':<6} | {'E Pred':<14} | {'E True':<8} | {'Err E(%)':<9} | {'Nu Pred':<14} | {'Nu True':<8} | {'Err Nu(%)':<9}")
print("-" * 110)

for region, mask in [("Base", base_mask), ("Red", red_mask), ("Blue", blue_mask)]:
    e_mean, e_std, nu_mean, nu_std = get_stats(mask, element_E_pred, element_nu_pred)
    e_true, nu_true = TRUE_PARAMS[region]
    e_err = abs(e_mean - e_true) / e_true * 100
    nu_err = abs(nu_mean - nu_true) / nu_true * 100
    print(f"{region:<8} | {len(mask):<6} | {e_mean:.4f}±{e_std:.4f} | {e_true:<8.2f} | {e_err:<9.2f} | {nu_mean:.4f}±{nu_std:.4f} | {nu_true:<8.2f} | {nu_err:<9.2f}")
    stats_dict[region] = {
        "E_mean": e_mean, "E_std": e_std, "E_true": e_true, "E_err": e_err,
        "Nu_mean": nu_mean, "Nu_std": nu_std, "Nu_true": nu_true, "Nu_err": nu_err
    }
print("-" * 110)

# ==============================================================================
# 12. 保存分析数据
# ==============================================================================
print("\n" + "=" * 80)
print("5. 保存最终分析数据 (error_analysis_final.npz)...")
print("=" * 80)

save_path = "error_analysis_final.npz"
np.savez(
    save_path,
    pred_ux=pred_ux, pred_uy=pred_uy,
    pred_sxx=pred_sxx, pred_syy=pred_syy, pred_sxy=pred_sxy,
    mesh_nodes=coors, mesh_elements=mesh_elements,
    element_E_pred=element_E_pred, element_nu_pred=element_nu_pred,
    # 新增真值保存
    element_E_true=element_E_true, element_nu_true=element_nu_true,
    mask_base=base_mask, mask_red=red_mask, mask_blue=blue_mask,
    param_stats=stats_dict,
    true_params=TRUE_PARAMS,
    disp_vmin=disp_vmin, disp_vmax=disp_vmax,
    stress_vmin=stress_vmin, stress_vmax=stress_vmax,
    E_vmin=E_vmin, E_vmax=E_vmax,
    nu_vmin=nu_vmin, nu_vmax=nu_vmax,
    disp_err_vmax=disp_err_vmax,
    stress_err_vmax=stress_err_vmax
)
print(f"✓ 数据已保存至: {save_path}")
print("✓ 所有任务完成！")