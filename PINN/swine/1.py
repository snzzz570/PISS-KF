import os
import numpy as np
import torch

import deepxde as dde
from deepxde.nn import activations
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from PIL import Image
import SimpleITK as sitk

# ==============================================================================
# 0. 批处理任务配置
# 修改说明：根据新路径结构 "Group/ID" 配置 sub_structure
# ==============================================================================
tasks = [
    # --- Group 2w ---
    {
        "sub_structure": "0w/7432375-4",
        "file_prefix": "7432375-4",
        "pixel_data_file": "7432375-4_pixel_data_ES1_to_ED11_idx9.npy",
        "flow_idx": 9,
        "output_root": "/code/zsn/mi_num8/material_final"
    },

]

# ==============================================================================
# 路径配置 (已修改为 mi_num8 相关路径)
# ==============================================================================
# 输入根路径 (Interpolation数据): /code/zsn/mi_num8/mesh_Interpolation
INPUT_ROOT = "/code/zsn/mi_num8/mesh_Interpolation"

# LGE图片路径: /code/zsn/mi_num8/our_sacr
# 代码逻辑中会拼接: LGE_ROOT + file_prefix + .png
LGE_ROOT = "/code/zsn/mi_num8/our_sacr"

# 变形场路径: /code/zsn/mi_num8/meshflow_output
# 代码逻辑中会拼接: FLOW_ROOT + sub_structure + mesh_flow_lag_2d.nii.gz
FLOW_ROOT = "/code/zsn/mi_num8/meshflow_output"


os.environ["DDEBACKEND"] = "pytorch"
if torch.cuda.is_available(): torch.cuda.set_device(0)
dde.config.set_random_seed(6001)

# ==============================================================================
# 1. 核心处理函数 (逻辑保持不变)
# ==============================================================================
def run_task(task):
    sub_structure = task["sub_structure"]
    file_prefix = task["file_prefix"]

    input_dir = os.path.join(INPUT_ROOT, sub_structure)
    output_dir = os.path.join(task["output_root"], sub_structure)
    os.makedirs(output_dir, exist_ok=True)

    lge_scar_path = os.path.join(LGE_ROOT, f"{file_prefix}.png")
    flow_file_path = os.path.join(FLOW_ROOT, sub_structure, "mesh_flow_lag_2d.nii.gz")
    pinn_data_path = os.path.join(input_dir, f"{file_prefix}_pinn_data.npy")
    pixel_data_path = os.path.join(input_dir, task["pixel_data_file"])

    # 加载数据
    data_pinn = np.load(pinn_data_path, allow_pickle=True).item()
    node_coors = data_pinn["coordinates"]
    mesh_elements = data_pinn["elements"]
    num_elements = len(mesh_elements)

    # 建立单元邻接关系
    edge_to_elements = {}
    for i, el in enumerate(mesh_elements):
        for j in range(3):
            edge = tuple(sorted((el[j], el[(j + 1) % 3])))
            if edge not in edge_to_elements: edge_to_elements[edge] = []
            edge_to_elements[edge].append(i)
    adj_indices = torch.tensor([els for els in edge_to_elements.values() if len(els) == 2]).cuda()

    # 运动场逻辑
    if os.path.exists(flow_file_path):
        itk_img = sitk.ReadImage(flow_file_path)
        flow_array = sitk.GetArrayFromImage(itk_img)
        target_flow = flow_array[task["flow_idx"]]
        node_disps = []
        for pt in node_coors:
            row = int(np.clip(np.round(pt[1]), 0, target_flow.shape[0] - 1))
            col = int(np.clip(np.round(pt[0]), 0, target_flow.shape[1] - 1))
            node_disps.append([target_flow[row, col, 1], target_flow[row, col, 0]])
        node_disps = np.array(node_disps)
    else:
        node_disps = data_pinn["displacements"]

    # 变形后坐标
    deformed_coors = node_coors + node_disps

    # 像素观测与掩码
    pix_dict = np.load(pixel_data_path, allow_pickle=True).item()
    pixel_coors = np.vstack([v['pixels'] for v in pix_dict.values() if len(v['pixels']) > 0])
    pixel_vals = np.vstack([v['displacements'] for v in pix_dict.values() if len(v['displacements']) > 0])
    element_centers = np.array([np.mean(node_coors[el], axis=0) for el in mesh_elements])

    if os.path.exists(lge_scar_path):
        lge_arr = np.array(Image.open(lge_scar_path).convert('L'))
        xi, yi = np.clip(element_centers[:, 0].astype(int), 0, lge_arr.shape[1] - 1), np.clip(
            element_centers[:, 1].astype(int), 0, lge_arr.shape[0] - 1)
        scar_mask = lge_arr[yi, xi] > 127
    else:
        scar_mask = np.zeros(num_elements, dtype=bool)

    centers_t = torch.tensor(element_centers, dtype=torch.float32).cuda()
    mask_t = torch.tensor(scar_mask, dtype=torch.bool).cuda()

    # --- 混合初始化逻辑 ---
    _E_init = torch.zeros((num_elements, 1)).cuda()
    _E_init[~mask_t] = 0.0
    _E_init[mask_t] = 28.0
    E_raw_vars = dde.Variable(_E_init + torch.randn_like(_E_init) * 0.1)
    nu_raw_vars = dde.Variable(torch.randn((num_elements, 1)).cuda() * 0.1)

    # --- 修改处 1：正常区域范围调整为 [10.0, 15.0] ---
    def map_params(indices):
        e_phys = torch.zeros((len(indices), 1)).cuda()
        e_raw = E_raw_vars[indices]
        # 正常区域回归 [10.0, 15.0]
        e_phys[~mask_t[indices]] = 12.0 + 5.0 * torch.sigmoid(e_raw[~mask_t[indices]])
        # 瘢痕区域继续使用 Softplus
        if mask_t.any():
            e_phys[mask_t[indices]] = torch.nn.functional.softplus(e_raw[mask_t[indices]])

        nu_phys = torch.zeros((len(indices), 1)).cuda()
        nu_raw = nu_raw_vars[indices]
        nu_phys[~mask_t[indices]] = 0.43 + 0.05 * torch.sigmoid(nu_raw[~mask_t[indices]])
        if mask_t.any():
            nu_phys[mask_t[indices]] = 0.45 + 0.03 * torch.sigmoid(nu_raw[mask_t[indices]])
        return e_phys, nu_phys

    def pde(x, y):
        ux, uy, nsxx, nsyy, ntxy = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]
        idx = torch.argmin(torch.cdist(x, centers_t), dim=1)
        E, nu = map_params(idx)
        exx, eyy = dde.grad.jacobian(ux, x, i=0, j=0), dde.grad.jacobian(uy, x, i=0, j=1)
        gxy = dde.grad.jacobian(ux, x, i=0, j=1) + dde.grad.jacobian(uy, x, i=0, j=0)
        fac = E / ((1 + nu) * (1 - 2 * nu))
        sxx_p = fac * ((1 - nu) * exx + nu * eyy)
        syy_p = fac * (nu * exx + (1 - nu) * eyy)
        txy_p = fac * ((1 - 2 * nu) / 2.0 * gxy)
        momentum_x = dde.grad.jacobian(nsxx, x, i=0, j=0) + dde.grad.jacobian(ntxy, x, i=0, j=1)
        momentum_y = dde.grad.jacobian(ntxy, x, i=0, j=0) + dde.grad.jacobian(nsyy, x, i=0, j=1)

        w = torch.ones_like(idx).float().reshape(-1, 1)
        w[mask_t[idx]] = 10.0

        E_all, nu_all = map_params(torch.arange(num_elements).cuda())
        tv_E = torch.mean(torch.abs(E_all[adj_indices[:, 0]] - E_all[adj_indices[:, 1]])) / 10.0
        tv_nu = torch.mean(torch.abs(nu_all[adj_indices[:, 0]] - nu_all[adj_indices[:, 1]])) / 0.03

        anchor_res = torch.tensor(0.0).cuda()
        if mask_t.any():
            anchor_res = (torch.mean(E_all[mask_t]) -30.0) ** 2

        extra_loss = (tv_E + tv_nu) * 0.1 + anchor_res * 1.0
        res_extra = extra_loss * torch.ones_like(momentum_x)

        return [momentum_x * w, momentum_y * w, (sxx_p - nsxx) * w, (syy_p - nsyy) * w, (txy_p - ntxy) * w, res_extra]

    class MPFNN(dde.nn.PFNN):
        def __init__(self, layer_sizes, second_layer_sizes, activation, kernel_initializer):
            super(MPFNN, self).__init__(layer_sizes, activation, kernel_initializer)
            self.firstFNN = dde.nn.PFNN(layer_sizes, activations.get(activation), kernel_initializer)
            self.secondFNN = dde.nn.PFNN(second_layer_sizes, activations.get(activation), kernel_initializer)
        def forward(self, inputs):
            return torch.cat((self.firstFNN(inputs), self.secondFNN(inputs)), dim=1)

    data = dde.data.PDE(dde.geometry.PointCloud(pixel_coors), pde,
                        [dde.PointSetBC(node_coors, node_disps, component=[0, 1]),
                         dde.PointSetBC(pixel_coors, pixel_vals, component=[0, 1])], anchors=pixel_coors)

    net = MPFNN([2] + [60] * 5 + [2], [2] + [60] * 5 + [3], "swish", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[E_raw_vars, nu_raw_vars])
    model.train(iterations=80000)

    final_E, final_nu = map_params(torch.arange(num_elements).cuda())
    final_E, final_nu = final_E.detach().cpu().numpy().flatten(), final_nu.detach().cpu().numpy().flatten()

    np.save(os.path.join(output_dir, f"{file_prefix}_E_results.npy"), final_E)
    np.save(os.path.join(output_dir, f"{file_prefix}_V_results.npy"), final_nu)
    print(f"数据保存完成: {output_dir}")

    # --- 去除可视化时的矩形边框 ---
    def plot_res(coords, vals, title, fname, vrange, cmap='jet'):
        plt.figure(figsize=(10, 8))
        tri = mtri.Triangulation(coords[:, 0], coords[:, 1], triangles=mesh_elements)
        pc = plt.tripcolor(tri, facecolors=vals, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        plt.triplot(tri, 'k-', linewidth=1, alpha=1.0)
        plt.colorbar(pc, label=title)
        plt.axis('equal')
        plt.axis('off')  # 关闭坐标轴和边框
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # 可视化
    plot_res(node_coors, final_E, "Young's Modulus E (kPa) - Ref", f"{file_prefix}_E_ref.png", [0, 35])
    plot_res(deformed_coors, final_E, "Young's Modulus E (kPa) - Def", f"{file_prefix}_E_def.png", [0, 35])
    plot_res(node_coors, final_nu, "Poisson's Ratio V (nu) - Ref", f"{file_prefix}_V_ref.png", [0.4, 0.48], cmap='bwr')
    plot_res(deformed_coors, final_nu, "Poisson's Ratio V (nu) - Def", f"{file_prefix}_V_def.png", [0.4, 0.48], cmap='bwr')

for t in tasks:
    print(f"正在处理任务: {t['file_prefix']}...")
    run_task(t)
    torch.cuda.empty_cache()