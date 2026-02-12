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
import re

# 如果有 GPU 则使用
if torch.cuda.is_available():
    torch.cuda.set_device(0)

dde.config.set_random_seed(2037)

# ==============================================================================
# 1. 读取仿真数据 & 预设参数
# ==============================================================================
data_path = "/code/zsn/Code3/Code1/sim_data/new_rectangular_block_pinn_data2.npy"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"错误: 找不到文件 {data_path}。请先运行仿真代码生成数据。")

data = np.load(data_path, allow_pickle=True).item()

if "elements" not in data:
    raise KeyError("数据文件缺失 'elements'！请确保仿真代码保存了网格信息。")

coors = data["coordinates"]
gt_disp = data["displacements"]
bnd_coors = data["boundary_coordinates"]
bnd_gt_disp = data["boundary_displacements"]
pressure_mag = data["pressure_top_mag"]
mesh_elements = data["elements"]

# 数据拼接
pde_pts = np.vstack((coors, bnd_coors))
pde_pts_disp = np.vstack((gt_disp, bnd_gt_disp))

# 几何定义
HEIGHT = 10.0
RED_X_RANGE = [3.0, 7.0]
RED_Y_RANGE = [4.0, 7.0]
BLUE_X_RANGE = [10.0, 14.0]
BLUE_Y_RANGE = [6.0, 9.0]

# --- 恢复您的原始设置 ---
s_scale = 1e5  # 固定值

# 统计位移特征
ux_std = np.std(pde_pts_disp[:, 0])
uy_std = np.std(pde_pts_disp[:, 1])

print(f"✓ 施加的压力: {pressure_mag:.2f} Pa")
print(f"✓ 应力缩放因子: {s_scale:.0e}")
print(f"✓ 原始网格信息: {len(coors)} Nodes, {len(mesh_elements)} Elements")


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
# 3. 输出变换 (硬约束)
# ==============================================================================
def output_transform(x, y):
    y_coord = x[:, 1:2]
    h = y_coord / HEIGHT

    # 位移
    u_net, v_net = y[:, 0:1], y[:, 1:2]
    u_phys = u_net * ux_std
    v_phys = v_net * uy_std
    u_final = u_phys * h
    v_final = v_phys * h

    # 应力
    sxx_net, syy_net, txy_net = y[:, 2:3], y[:, 3:4], y[:, 4:5]
    sxx_phys = sxx_net * s_scale
    syy_phys = syy_net * s_scale
    txy_phys = txy_net * s_scale

    target_syy = -pressure_mag
    target_txy = 0.0

    syy_final = syy_phys * (1.0 - h) + target_syy * h
    txy_final = txy_phys * (1.0 - h) + target_txy * h
    sxx_final = sxx_phys

    return torch.cat([u_final, v_final, sxx_final, syy_final, txy_final], dim=1)


# ==============================================================================
# 4. 物理场定义 (平滑掩膜)
# ==============================================================================
def smooth_mask(coord, coord_min, coord_max, width=0.3):
    left = torch.sigmoid((coord - coord_min) / width)
    right = torch.sigmoid((coord_max - coord) / width)
    return left * right


# 可训练参数
E_base_var = dde.Variable(0.0)
nu_base_var = dde.Variable(0.0)
E_red_var = dde.Variable(0.0)
nu_red_var = dde.Variable(0.0)
E_blue_var = dde.Variable(0.0)
nu_blue_var = dde.Variable(0.0)


def stress(x, y):
    ux, uy = y[:, 0:1], y[:, 1:2]
    exx = dde.grad.jacobian(ux, x, i=0, j=0)
    eyy = dde.grad.jacobian(uy, x, i=0, j=1)
    gxy = dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)

    x_loc, y_loc = x[:, 0:1], x[:, 1:2]

    # 平滑掩膜
    mask_red = smooth_mask(x_loc, RED_X_RANGE[0], RED_X_RANGE[1], 0.3) * \
               smooth_mask(y_loc, RED_Y_RANGE[0], RED_Y_RANGE[1], 0.3)
    mask_blue = smooth_mask(x_loc, BLUE_X_RANGE[0], BLUE_X_RANGE[1], 0.3) * \
                smooth_mask(y_loc, BLUE_Y_RANGE[0], BLUE_Y_RANGE[1], 0.3)

    total_special = mask_red + mask_blue
    mask_base = torch.clamp(1.0 - total_special, min=0.0)

    # 归一化
    total_mask = mask_base + mask_red + mask_blue + 1e-10
    mask_base = mask_base / total_mask
    mask_red = mask_red / total_mask
    mask_blue = mask_blue / total_mask

    # 参数映射
    def map_param(var, min_val, max_val):
        return min_val + (torch.tanh(var) + 1.0) / 2.0 * (max_val - min_val)

    nu_base = map_param(nu_base_var, 0.01, 0.49)
    nu_red = map_param(nu_red_var, 0.01, 0.49)
    nu_blue = map_param(nu_blue_var, 0.01, 0.49)

    E_base = map_param(E_base_var, 0.1, 5.0) * s_scale
    E_red = map_param(E_red_var, 0.1, 5.0) * s_scale
    E_blue = map_param(E_blue_var, 0.1, 5.0) * s_scale

    nu_field = (nu_base * mask_base + nu_red * mask_red + nu_blue * mask_blue)
    E_field = (E_base * mask_base + E_red * mask_red + E_blue * mask_blue)

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

    momentum_x = (sxx_x + txy_y) / s_scale
    momentum_y = (txy_x + syy_y) / s_scale

    sxx_phys, syy_phys, txy_phys, _ = stress(x, y)

    # 区域加权
    x_loc, y_loc = x[:, 0:1], x[:, 1:2]
    is_blue = (x_loc > BLUE_X_RANGE[0]) & (x_loc < BLUE_X_RANGE[1]) & \
              (y_loc > BLUE_Y_RANGE[0]) & (y_loc < BLUE_Y_RANGE[1])
    weight = 1.0 + 5.0 * is_blue.float()

    stress_xx_res = (sxx_phys - Nsxx) / s_scale * weight
    stress_yy_res = (syy_phys - Nsyy) / s_scale * weight
    stress_xy_res = (txy_phys - Ntxy) / s_scale * weight

    return [momentum_x, momentum_y, stress_xx_res, stress_yy_res, stress_xy_res]


# ==============================================================================
# 6. 训练过程
# ==============================================================================
geom = dde.geometry.PointCloud(points=pde_pts)
losses_data = [dde.PointSetBC(pde_pts, pde_pts_disp, component=[0, 1])]
data_obj = dde.data.PDE(geom, pde, losses_data, anchors=pde_pts)

net = MPFNN([2] + [45] * 5 + [2], [2] + [45] * 5 + [3], "swish", "Glorot normal")
net.apply_output_transform(output_transform)
model = dde.Model(data_obj, net)

vars_list = [E_base_var, nu_base_var, E_red_var, nu_red_var, E_blue_var, nu_blue_var]
fname_var = "material_params_smooth_boundary.dat"
variable_recorder = dde.callbacks.VariableValue(vars_list, period=1000, filename=fname_var)

print("\n开始训练...")
model.compile("adam", lr=1e-3, decay=["step", 15000, 0.5],
              loss_weights=[1, 1, 1, 1, 1, 10],
              external_trainable_variables=vars_list)

losshistory, train_state = model.train(iterations=100000, callbacks=[variable_recorder], display_every=1000)
dde.saveplot(losshistory, train_state, issave=True, isplot=False)

# ==============================================================================
# 7. 可视化 Part 1: 训练曲线 & 参数反演
# ==============================================================================
print("\n" + "=" * 80)
print("1. 生成训练监控曲线...")
print("=" * 80)

# (1) Loss 曲线
plt.figure(figsize=(6, 4))
loss_train = np.sum(losshistory.loss_train, axis=1)
plt.semilogy(loss_train, 'b-', label='Total Loss')
plt.xlabel('Iteration');
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("Fig_Loss.png", dpi=300)
print("   -> 已保存: Fig_Loss.png")

# (2) 参数收敛曲线
if os.path.exists(fname_var):
    lines = open(fname_var, "r").readlines()
    records = []
    steps = []
    for i, line in enumerate(lines):
        steps.append(i * 1000)
        vals_str = min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len)
        vals = np.fromstring(vals_str, sep=",")
        records.append(vals)
    records = np.array(records)


    # 还原物理值
    def map_val_E(v):
        return 0.1 + (np.tanh(v) + 1.0) / 2.0 * 4.9  # [0.1, 5.0]


    def map_val_nu(v):
        return 0.01 + (np.tanh(v) + 1.0) / 2.0 * 0.48  # [0.01, 0.49]


    # 乘以 s_scale
    E_base_hist = map_val_E(records[:, 0]) * s_scale
    nu_base_hist = map_val_nu(records[:, 1])
    E_red_hist = map_val_E(records[:, 2]) * s_scale
    nu_red_hist = map_val_nu(records[:, 3])
    E_blue_hist = map_val_E(records[:, 4]) * s_scale
    nu_blue_hist = map_val_nu(records[:, 5])

    TRUE_PARAMS = {
        "Base": (75000.0, 0.45),
        "Red": (105000.0, 0.49),
        "Blue": (45000.0, 0.40)
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(steps, E_base_hist, 'k-', label='Base')
    ax1.plot(steps, E_red_hist, 'r-', label='Red')
    ax1.plot(steps, E_blue_hist, 'b-', label='Blue')
    ax1.axhline(TRUE_PARAMS["Base"][0], color='k', linestyle='--')
    ax1.axhline(TRUE_PARAMS["Red"][0], color='r', linestyle='--')
    ax1.axhline(TRUE_PARAMS["Blue"][0], color='b', linestyle='--')
    ax1.set_title("Young's Modulus E");
    ax1.legend();
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, nu_base_hist, 'k-', label='Base')
    ax2.plot(steps, nu_red_hist, 'r-', label='Red')
    ax2.plot(steps, nu_blue_hist, 'b-', label='Blue')
    ax2.axhline(TRUE_PARAMS["Base"][1], color='k', linestyle='--')
    ax2.axhline(TRUE_PARAMS["Red"][1], color='r', linestyle='--')
    ax2.axhline(TRUE_PARAMS["Blue"][1], color='b', linestyle='--')
    ax2.set_title("Poisson's Ratio Nu");
    ax2.legend();
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig_Param_Convergence.png", dpi=300)
    print("   -> 已保存: Fig_Param_Convergence.png")

# ==============================================================================
# 8. 可视化 Part 2: 高级 FEM 风格网格展示
# ==============================================================================
print("\n" + "=" * 80)
print("2. 生成 FEM 风格可视化 (原始网格 + 变形网格)...")
print("=" * 80)

# 1. 准备原始网格
triang_orig = mtri.Triangulation(coors[:, 0], coors[:, 1], triangles=mesh_elements)

# 2. 准备预测数据
y_pred_mesh = model.predict(coors)
pred_ux, pred_uy = y_pred_mesh[:, 0], y_pred_mesh[:, 1]
pred_sxx, pred_syy, pred_sxy = y_pred_mesh[:, 2], y_pred_mesh[:, 3], y_pred_mesh[:, 4]

# 3. 准备真值数据
true_ux = data["displacements"][:, 0]
true_uy = data["displacements"][:, 1]
true_sxx = data["stress_xx"].flatten()
true_syy = data["stress_yy"].flatten()
true_sxy = data["stress_xy"].flatten()

# 4. 准备变形网格 (Deformed Mesh)
# 设置变形放大系数 (Scale Factor)，如果是1.0就是真实比例
SCALE_FACTOR = 1.0

# 预测的变形网格
x_def_pred = coors[:, 0] + pred_ux * SCALE_FACTOR
y_def_pred = coors[:, 1] + pred_uy * SCALE_FACTOR
triang_def_pred = mtri.Triangulation(x_def_pred, y_def_pred, triangles=mesh_elements)

# 真值的变形网格 (用于对比)
x_def_true = coors[:, 0] + true_ux * SCALE_FACTOR
y_def_true = coors[:, 1] + true_uy * SCALE_FACTOR
triang_def_true = mtri.Triangulation(x_def_true, y_def_true, triangles=mesh_elements)


# -------------------------------------------------------------------------
# 通用绘图函数 (支持原始网格和变形网格)
# -------------------------------------------------------------------------
def save_single_field(triang_obj, values, title, filename, unit, vmin=None, vmax=None, cmap='jet', show_mesh=True):
    plt.figure(figsize=(7, 5), facecolor='white')
    if vmin is None: vmin = np.min(values)
    if vmax is None: vmax = np.max(values)
    levels = np.linspace(vmin, vmax, 25)
    cnt = plt.tricontourf(triang_obj, values, levels=levels, cmap=cmap, extend='both')

    if show_mesh:
        plt.triplot(triang_obj, 'k-', linewidth=1.0, alpha=0.6)  # 加粗网格线

    cbar = plt.colorbar(cnt, fraction=0.046, pad=0.04)
    cbar.set_label(unit, fontsize=12)
    cbar.formatter.set_powerlimits((-2, 2))
    plt.title(title, fontsize=14)
    plt.axis('equal');
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> 已保存: {filename}")


def save_all_combinations(val_true, val_pred, name_short, name_full, unit):
    # 统一色标
    g_min = min(np.min(val_true), np.min(val_pred))
    g_max = max(np.max(val_true), np.max(val_pred))

    # 1. 原始网格对比 (Original Mesh)
    save_single_field(triang_orig, val_true, f"GT (Orig): {name_full}",
                      f"Orig_GT_{name_short}.png", unit, vmin=g_min, vmax=g_max)
    save_single_field(triang_orig, val_pred, f"Pred (Orig): {name_full}",
                      f"Orig_Pred_{name_short}.png", unit, vmin=g_min, vmax=g_max)

    # 2. 变形网格对比 (Deformed Mesh)
    save_single_field(triang_def_true, val_true, f"GT (Deformed): {name_full}",
                      f"Def_GT_{name_short}.png", unit, vmin=g_min, vmax=g_max)
    save_single_field(triang_def_pred, val_pred, f"Pred (Deformed): {name_full}",
                      f"Def_Pred_{name_short}.png", unit, vmin=g_min, vmax=g_max)


# -------------------------------------------------------------------------
# 执行批量绘图
# -------------------------------------------------------------------------
print("   -> 生成位移对比图 (原始 & 变形)...")
save_all_combinations(true_ux, pred_ux, "Ux", "Disp X", "m")
save_all_combinations(true_uy, pred_uy, "Uy", "Disp Y", "m")

print("   -> 生成应力对比图 (原始 & 变形)...")
save_all_combinations(true_sxx, pred_sxx, "Sxx", "Stress XX", "Pa")
save_all_combinations(true_syy, pred_syy, "Syy", "Stress YY", "Pa")
save_all_combinations(true_sxy, pred_sxy, "Sxy", "Stress XY", "Pa")

# -------------------------------------------------------------------------
# E场重构 (在原始网格 & 变形网格上)
# -------------------------------------------------------------------------
print("\n3. 重构并绘制 E 场 (原始 & 变形)...")

if len(records) > 0:
    last_vec = records[-1]
    E_base_final = map_val_E(last_vec[0]) * s_scale
    E_red_final = map_val_E(last_vec[2]) * s_scale
    E_blue_final = map_val_E(last_vec[4]) * s_scale

    # 构造单元颜色数组
    element_E_colors = []
    for el in mesh_elements:
        centroid = np.mean(coors[el], axis=0)
        cx, cy = centroid[0], centroid[1]
        if (RED_X_RANGE[0] <= cx <= RED_X_RANGE[1]) and (RED_Y_RANGE[0] <= cy <= RED_Y_RANGE[1]):
            element_E_colors.append(E_red_final)
        elif (BLUE_X_RANGE[0] <= cx <= BLUE_X_RANGE[1]) and (BLUE_Y_RANGE[0] <= cy <= BLUE_Y_RANGE[1]):
            element_E_colors.append(E_blue_final)
        else:
            element_E_colors.append(E_base_final)
    element_E_colors = np.array(element_E_colors)


    # 绘图函数: tripcolor
    def save_E_plot(triang_obj, title, filename):
        plt.figure(figsize=(7, 5), facecolor='white')
        tpc = plt.tripcolor(triang_obj, facecolors=element_E_colors, cmap='jet', edgecolors='k', linewidth=1.0)
        cbar = plt.colorbar(tpc, fraction=0.046, pad=0.04)
        cbar.set_label("Young's Modulus E (Pa)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.axis('equal');
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> 已保存: {filename}")


    # 1. 原始网格上的 E
    save_E_plot(triang_orig, "Predicted E (Original Mesh)", "Orig_Pred_E_Field.png")

    # 2. 变形网格上的 E (新增需求)
    save_E_plot(triang_def_pred, "Predicted E (Deformed Mesh)", "Def_Pred_E_Field.png")

# ==============================================================================
# 9. 定量误差表
# ==============================================================================
print("\n" + "=" * 80)
print("3. 生成定量误差表...")
print("=" * 80)

preds = {
    "Base": (E_base_final, nu_base_hist[-1]),
    "Red": (E_red_final, nu_red_hist[-1]),
    "Blue": (E_blue_final, nu_blue_hist[-1])
}

print(f"{'Region':<8} | {'Pred E (Pa)':<12} | {'True E (Pa)':<12} | {'Err E (%)':<10} | "
      f"{'Pred Nu':<9} | {'True Nu':<9} | {'Err Nu (%)':<10}")
print("-" * 100)

for region in ["Base", "Red", "Blue"]:
    pe, pn = preds[region]
    te, tn = TRUE_PARAMS[region]
    err_e = abs(pe - te) / te * 100
    err_nu = abs(pn - tn) / tn * 100
    print(f"{region:<8} | {pe:<12.1f} | {te:<12.0f} | {err_e:<10.2f} | "
          f"{pn:<9.4f} | {tn:<9.4f} | {err_nu:<10.2f}")
print("-" * 100)

np.savez(
    'error_analysis_final.npz',
    pred_ux=pred_ux, pred_uy=pred_uy,
    pred_sxx=pred_sxx, pred_syy=pred_syy,
    mesh_nodes=coors, mesh_elements=mesh_elements,
    param_preds=preds
)

print("\n✓ 所有任务已完成！变形网格可视化已添加。")
print("=" * 80)