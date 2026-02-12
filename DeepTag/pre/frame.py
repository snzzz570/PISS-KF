#(6)

import numpy as np
import cv2
import os
import SimpleITK as sitk

def read_cfg_simple(cfg_path):
    config = {}
    with open(cfg_path, 'r') as file:
        for line in file:
            if ': ' in line:
                key, value = line.strip().split(': ')
                config[key] = value
    return config


def save_slices_from_nii(nii_path, save_path):
    # 读取.nii文件
    name = os.path.basename(nii_path).split('_')[1]
    nii_img = sitk.ReadImage(nii_path)
    img_data = sitk.GetArrayFromImage(nii_img)

    original_spacing = nii_img.GetSpacing()
    original_direction = nii_img.GetDirection()
    original_origin = nii_img.GetOrigin()
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    processed_slices = set()

    # 遍历每个深度方向的切片
    for i in range(img_data.shape[0]):
        # 提取每个切片
        print(img_data.shape)
        frame = img_data[ i, :, :]
        print(frame.shape)


        # if frame.tobytes() in processed_slices:
        #     print(f"Slice {i+1} is the same as a previous slice. Stopping the loop.")
        #     print(nii_path)
        #     break

        # # 将切片添加到已处理过的集合中
        # processed_slices.add(frame.tobytes())
        crop_size = (256, 256)
        if frame.shape[0] < crop_size[0] or frame.shape[1] < crop_size[1]:
            pad_height = max(0, crop_size[0] - frame.shape[0])
            pad_width = max(0, crop_size[1] - frame.shape[1])
            frame = np.pad(frame , ((0, pad_height), (0, pad_width)), mode='constant')

        # 计算裁剪的起始位置
        start_x = (frame.shape[0] - crop_size[0]) // 2
        start_y = (frame.shape[1] - crop_size[1]) // 2

        # 计算裁剪的结束位置
        end_x = start_x + crop_size[0]
        end_y = start_y + crop_size[1]


        frame = frame[start_x:end_x, start_y:end_y]
        print(frame.shape)

        frame_img = sitk.GetImageFromArray(frame)
        # 重新设置空间信息
        frame_img.SetSpacing((original_spacing[0], original_spacing[1]))
        frame_img.SetDirection((1.0, 0.0,
                         0.0, 1.0))
        
        
        # 保存切片
        file_path_3d = os.path.join(save_path, f'slice_{i+1}_'+name+'.nii.gz')
        sitk.WriteImage(frame_img, file_path_3d)
        

#dataroot = "./database/test"

dataroot ="/data2/zsn/Dataset/ACDC/database/training"
for root, dirs, files in os.walk(dataroot):
    for file in files:
        if file.endswith("_gt.nii.gz"):
            file_path = os.path.join(root, file)
            save_path = os.path.join(root, "slicesgt")
            save_slices_from_nii(file_path, save_path)