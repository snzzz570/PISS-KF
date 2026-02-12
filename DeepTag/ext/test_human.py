import sys
import os
import numpy as np
import torch
import time
from DeepTag.ME_nets.Deeptagnet import Lagrangian_motion_estimate_net
import SimpleITK as sitk
from DeepTag.data_set.utils import TagDataset1
from tqdm import tqdm

# --- 1. 设备配置 (强制 CPU) ---
device = torch.device('cpu')
print(f"⚙️ 当前运行设备: {device}")


# ---------------------------

def load_dec_weights(model, weights):
    print('Resuming net weights from {} ...'.format(weights))
    # 加载权重时映射到 CPU
    w_dict = torch.load(weights, map_location=device)
    model.load_state_dict(w_dict, strict=True)
    return model


def extract_path_info_new_structure(filepath):
    """
    针对新的 Human_data 结构进行解析
    结构范例: .../nii_output/mi/5277697-5.nii.gz
    返回: group_name (mi/myo), file_id
    """
    # 统一路径分隔符
    filepath = filepath.replace('\\', '/')

    # 获取文件名
    filename = os.path.basename(filepath)  # "5277697-5.nii.gz"

    # 获取 ID
    file_id = filename.replace('.nii.gz', '').replace('.nii', '')  # "5277697-5"

    # 获取父目录 (即组名 mi 或 myo)
    parent_dir = os.path.dirname(filepath)
    group_name = os.path.basename(parent_dir)  # "mi" 或 "myo"

    # 简单校验
    if group_name not in ['mi', 'myo']:
        # 如果不是直接在 mi/myo 下，可能是在更深或更浅的目录，做一个兼容处理
        # 假设父目录就是组名
        pass

    return group_name, file_id


def is_valid_nii_file(filepath):
    """检查nii.gz文件是否有效"""
    try:
        if not os.path.exists(filepath): return False
        if os.path.getsize(filepath) == 0: return False

        # 简单读取测试
        # cine_image = sitk.ReadImage(filepath)
        # if cine_image.GetSize()[0] == 0: return False

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
    # 注意：TagDataset1 通常会递归搜索目录，所以只要 data_root 指向 nii_output 即可
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

    print(f"开始处理，数据源: {data_root}")

    for i, data in tqdm(enumerate(test_set_loader), total=len(test_set_loader), desc="Processing (CPU)"):
        name, cine = data
        input_path = name[0]

        # 解析路径信息
        group_name, file_id = extract_path_info_new_structure(input_path)

        # 构造输出路径: Output/mi/5277697-5/...
        tgt_folder = os.path.join(dst_root, group_name, file_id)
        os.makedirs(tgt_folder, exist_ok=True)

        print(f"\n处理: [{group_name}] -> [{file_id}]")

        # 检查有效性
        if not is_valid_nii_file(input_path):
            print(f"  -> 无效文件或无法读取，跳过。")
            continue

        try:
            # 移动数据到 CPU
            cine = cine.to(device).float()

            x = cine[:, 1:, ::]
            shape = x.shape
            seq_length, height, width = shape[1], shape[2], shape[3]
            y = cine[:, 0:-1, ::]

            x = x.contiguous().view(-1, 1, height, width)
            y = y.contiguous().view(-1, 1, height, width)

            with torch.no_grad():
                start_t = time.time()
                # 网络推断
                val_registered_cine1, val_registered_cine2, val_registered_cine_lag, \
                    val_flow_param, val_deformation_matrix, val_deformation_matrix_neg, \
                    val_deformation_matrix_lag = net(y, x)

                # 计算拉格朗日流
                val_neg_lag_flow = net.lag_flow(val_deformation_matrix_neg)

                end_t = time.time()

            if i != 0: sample_times.append(end_t - start_t)

            # 后处理 & 转 Numpy
            val_inf_cine = val_registered_cine1[:, 0, ::].cpu().detach().numpy()
            val_registered_cine = val_registered_cine_lag[:, 0, ::].cpu().detach().numpy()

            val_deformation_matrix_lag2d = val_deformation_matrix_lag.permute(0, 2, 3, 1).cpu().detach().numpy()
            val_deformation_matrix_lag0 = val_deformation_matrix_lag[:, 0, ::].cpu().detach().numpy()
            val_deformation_matrix_lag1 = val_deformation_matrix_lag[:, 1, ::].cpu().detach().numpy()

            val_neg_lag_flow_2d = val_neg_lag_flow.permute(0, 2, 3, 1).cpu().detach().numpy()
            val_neg_lag_flow_x = val_neg_lag_flow[:, 0, ::].cpu().detach().numpy()
            val_neg_lag_flow_y = val_neg_lag_flow[:, 1, ::].cpu().detach().numpy()

            # 读取原始空间信息，确保输出的nii与原图对齐
            raw_nii = sitk.ReadImage(input_path)
            spacing = raw_nii.GetSpacing()
            origin = raw_nii.GetOrigin()
            direction = raw_nii.GetDirection()

            def save_result_nii(data, filename):
                img = sitk.GetImageFromArray(data)
                img.SetSpacing(spacing)
                img.SetOrigin(origin)
                img.SetDirection(direction)
                sitk.WriteImage(img, os.path.join(tgt_folder, filename))

            # 保存结果
            save_result_nii(val_inf_cine, 'inf_cine.nii.gz')
            save_result_nii(val_registered_cine, 'registered_cine.nii.gz')

            save_result_nii(val_deformation_matrix_lag0, 'deformation_matrix_x.nii.gz')
            save_result_nii(val_deformation_matrix_lag1, 'deformation_matrix_y.nii.gz')
            save_result_nii(val_deformation_matrix_lag2d, 'deformation_matrix_2d.nii.gz')

            save_result_nii(val_neg_lag_flow_x, 'mesh_flow_lag_x.nii.gz')
            save_result_nii(val_neg_lag_flow_y, 'mesh_flow_lag_y.nii.gz')
            save_result_nii(val_neg_lag_flow_2d, 'mesh_flow_lag_2d.nii.gz')

            val_inf_flow_2d = val_deformation_matrix.permute(0, 2, 3, 1).cpu().detach().numpy()
            val_inf_flow_x = val_deformation_matrix[:, 0, ::].cpu().detach().numpy()
            val_inf_flow_y = val_deformation_matrix[:, 1, ::].cpu().detach().numpy()

            save_result_nii(val_inf_flow_x, 'inf_flow_x.nii.gz')
            save_result_nii(val_inf_flow_y, 'inf_flow_y.nii.gz')
            save_result_nii(val_inf_flow_2d, 'inf_flow_2d.nii.gz')

            print(f"  -> 保存成功")
            processed_count += 1

        except Exception as e:
            print(f"  [运行错误] {file_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n全部完成。共处理 {processed_count} 个样本。")
    if sample_times:
        print(f"平均时间: {np.mean(sample_times):.4f} s")


if __name__ == '__main__':
    # --- 核心路径配置 ---
    # 输入路径：上一轮生成的 nii_output 文件夹
    # data_path_root = r"C:\Users\zsn\Desktop\Humandata\Human_data\nii_output"
    #
    # # 输出路径：新建一个 meshflow_output
    # dst_root = r"C:\Users\zsn\Desktop\Humandata\Human_data\meshflow_output"

    # data_path_root = r"C:\Users\zsn\Desktop\Humandata\Human_data_new\nii_output"
    #
    # # 输出路径：新建一个 meshflow_output
    # dst_root = r"C:\Users\zsn\Desktop\Humandata\Human_data_new\meshflow_output"

    data_path_root = r"C:\Users\zsn\Desktop\Humandata\Humandata_new2\nii_output"

    # 输出路径：新建一个 meshflow_output
    dst_root = r"C:\Users\zsn\Desktop\Humandata\Humandata_new2\meshflow_output"


    # 模型路径 (保持不变，或根据实际情况修改)
    test_model_path = './128'

    # 网络参数
    vol_size = (128, 128)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]

    print(f"Data Root: {data_path_root}")
    print(f"Output Root: {dst_root}")

    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    test_Cardiac_Tagging_ME_net(
        net=net,
        data_root=data_path_root,
        model_path=test_model_path,
        dst_root=dst_root,
        case='choosed'
    )