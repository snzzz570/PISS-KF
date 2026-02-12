import sys
import os
import numpy as np
import torch
import time
from DeepTag.ME_nets.Deeptagnet import Lagrangian_motion_estimate_net
import SimpleITK as sitk
from DeepTag.data_set.utils import TagDataset1, ssim, psnr, dice, compute_metrics_all_classes
from tqdm import tqdm
import re
import shutil

# --- 修改点 1：强制使用 CPU ---
# 注释掉 CUDA_VISIBLE_DEVICES，或者将其设为空
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cpu')
print(f"⚙️ 当前运行设备: {device}")


# ---------------------------


def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    # --- 修改点 2：加载权重时映射到 CPU ---
    # 如果权重是在 GPU 上训练保存的，加载到 CPU 必须加 map_location
    w_dict = torch.load(weights, map_location=device)
    model.load_state_dict(w_dict, strict=True)
    return model


def extract_path_info_new_structure(filepath):
    """
    针对结构: .../16/8674925-8/8674925-4.nii.gz 进行解析
    """
    # 统一路径分隔符
    filepath = filepath.replace('\\', '/')

    # 获取文件名和目录结构
    filename = os.path.basename(filepath)  # "8674925-4.nii.gz"

    # 去掉 .nii.gz 后缀作为文件ID
    file_id = filename.replace('.nii.gz', '').replace('.nii', '')  # "8674925-4"

    parent_dir = os.path.dirname(filepath)  # ".../16/8674925-8"
    group_id = os.path.basename(parent_dir)  # "8674925-8"

    grandparent_dir = os.path.dirname(parent_dir)  # ".../16"
    subject_id = os.path.basename(grandparent_dir)  # "16"

    # 简单的校验
    if not subject_id or not group_id:
        return "unknown_subj", "unknown_group", file_id

    return subject_id, group_id, file_id


def is_valid_nii_file(filepath):
    """检查nii.gz文件是否有效"""
    try:
        if not os.path.exists(filepath): return False
        if os.path.getsize(filepath) == 0: return False

        # 简单读取测试
        cine_image = sitk.ReadImage(filepath)
        if cine_image.GetSize()[0] == 0: return False

        return True
    except Exception as e:
        print(f"  [文件损坏] {filepath}: {e}")
        return False


def test_Cardiac_Tagging_ME_net(net,
                                data_root,
                                model_path,
                                dst_root,
                                case='proposed'):
    # 加载测试数据集
    test_dataset = TagDataset1(data_path=data_root, data_type='test')
    test_batch_size = 1
    test_set_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )

    os.makedirs(dst_root, exist_ok=True)

    model_file = '/pro_new_model.pth' if case == 'proposed' else '/end_model.pth'
    model_path = model_path + model_file

    ME_model = load_dec_weights(net, model_path)
    ME_model = ME_model.to(device)
    ME_model.eval()

    sample_times = []
    processed_count = 0

    for i, data in tqdm(enumerate(test_set_loader), total=len(test_set_loader), desc="Processing (CPU)"):
        name, cine = data
        input_path = name[0]

        subject_id, group_id, file_id = extract_path_info_new_structure(input_path)

        print(f"\n处理: [{subject_id}] -> [{group_id}] -> [{file_id}]")

        tgt_folder = os.path.join(dst_root, subject_id, group_id, file_id)
        os.makedirs(tgt_folder, exist_ok=True)

        # 检查有效性
        if not is_valid_nii_file(input_path):
            print(f"  -> 无效文件，跳过计算。")
            continue

        try:
            # 移动数据到 CPU (device='cpu')
            cine = cine.to(device).float()

            x = cine[:, 1:, ::]
            shape = x.shape
            seq_length, height, width = shape[1], shape[2], shape[3]
            y = cine[:, 0:-1, ::]

            x = x.contiguous().view(-1, 1, height, width)
            y = y.contiguous().view(-1, 1, height, width)

            # 这里的 t 也要确保在 device 上
            # (原代码虽然这行被注释了或未显示完整，但为了保险起见，如果有定义 t，也应该是 .to(device))
            # t = torch.ones(seq_length, 1).to(device)

            with torch.no_grad():
                start_t = time.time()
                # 网络推断
                val_registered_cine1, val_registered_cine2, val_registered_cine_lag, \
                    val_flow_param, val_deformation_matrix, val_deformation_matrix_neg, \
                    val_deformation_matrix_lag = net(y, x)

                # ========== 计算neg_inf_flow的拉格朗日流（用于网格可视化） ==========
                # 使用网络内部的lag_flow模块对neg_inf_flow进行拉格朗日累积
                val_neg_lag_flow = net.lag_flow(val_deformation_matrix_neg)

                end_t = time.time()

            if i != 0: sample_times.append(end_t - start_t)

            # 后处理
            val_inf_cine = val_registered_cine1[:, 0, ::].cpu().detach().numpy()
            val_registered_cine = val_registered_cine_lag[:, 0, ::].cpu().detach().numpy()

            # 反向位移场（用于图像配准）
            val_deformation_matrix_lag2d = val_deformation_matrix_lag.permute(0, 2, 3, 1).cpu().detach().numpy()
            val_deformation_matrix_lag0 = val_deformation_matrix_lag[:, 0, ::].cpu().detach().numpy()
            val_deformation_matrix_lag1 = val_deformation_matrix_lag[:, 1, ::].cpu().detach().numpy()

            # ========== 正向网格运动场（用于网格可视化） ==========
            # 从neg_inf_flow的拉格朗日流计算得到
            val_neg_lag_flow_2d = val_neg_lag_flow.permute(0, 2, 3, 1).cpu().detach().numpy()
            val_neg_lag_flow_x = val_neg_lag_flow[:, 0, ::].cpu().detach().numpy()
            val_neg_lag_flow_y = val_neg_lag_flow[:, 1, ::].cpu().detach().numpy()

            # 读取原始空间信息
            raw_nii = sitk.ReadImage(input_path)
            spacing = raw_nii.GetSpacing()
            origin = raw_nii.GetOrigin()
            direction = raw_nii.GetDirection()

            def save_result_nii(data, filename, is_vector=False):
                img = sitk.GetImageFromArray(data)
                img.SetSpacing(spacing)
                img.SetOrigin(origin)
                img.SetDirection(direction)
                sitk.WriteImage(img, os.path.join(tgt_folder, filename))

            # 保存所有结果
            save_result_nii(val_inf_cine, 'inf_cine.nii.gz')
            save_result_nii(val_registered_cine, 'registered_cine.nii.gz')

            # 保存反向位移场（用于图像配准warping）
            save_result_nii(val_deformation_matrix_lag0, 'deformation_matrix_x.nii.gz')
            save_result_nii(val_deformation_matrix_lag1, 'deformation_matrix_y.nii.gz')
            save_result_nii(val_deformation_matrix_lag2d, 'deformation_matrix_2d.nii.gz')

            # 保存正向网格运动场（用于网格可视化，从neg_inf_flow的拉格朗日流计算）
            save_result_nii(val_neg_lag_flow_x, 'mesh_flow_lag_x.nii.gz')
            save_result_nii(val_neg_lag_flow_y, 'mesh_flow_lag_y.nii.gz')
            save_result_nii(val_neg_lag_flow_2d, 'mesh_flow_lag_2d.nii.gz')

            # Flow
            val_inf_flow_2d = val_deformation_matrix.permute(0, 2, 3, 1).cpu().detach().numpy()
            val_inf_flow_x = val_deformation_matrix[:, 0, ::].cpu().detach().numpy()
            val_inf_flow_y = val_deformation_matrix[:, 1, ::].cpu().detach().numpy()

            save_result_nii(val_inf_flow_x, 'inf_flow_x.nii.gz')
            save_result_nii(val_inf_flow_y, 'inf_flow_y.nii.gz')
            save_result_nii(val_inf_flow_2d, 'inf_flow_2d.nii.gz')

            print(f"  -> 成功保存至: {tgt_folder}")
            processed_count += 1

        except Exception as e:
            print(f"  [运行错误] {file_id}: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误栈，方便调试

    print(f"\n完成。共处理 {processed_count} 个样本。")
    if sample_times:
        print(f"平均时间: {np.mean(sample_times):.4f} s")


if __name__ == '__main__':
    # 配置路径
    data_path_root = r"C:\Users\zsn\Desktop\pigdataMI\es_first_nii-output"
    dst_root = r"C:\Users\zsn\Desktop\pigdataMI\es_first_meshflow_output"
    test_model_path = './128'

    # 网络参数
    vol_size = (128, 128)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]

    print(f"数据源: {data_path_root}")
    print(f"输出地: {dst_root}")

    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    test_Cardiac_Tagging_ME_net(
        net=net,
        data_root=data_path_root,
        model_path=test_model_path,
        dst_root=dst_root,
        case='choosed'
    )