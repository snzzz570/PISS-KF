#(4)

import numpy as np
import cv2
import os
import SimpleITK as sitk

def save_slices_from_nii(nii_path, save_path):
    # 读取.nii文件
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
    for i in range(img_data.shape[1]):
        # 提取每个切片
        print(img_data.shape)
        slice_3d = img_data[:, i, :, :]
        print(slice_3d.shape)


        # if slice_3d.tobytes() in processed_slices:
        #     print(f"Slice {i+1} is the same as a previous slice. Stopping the loop.")
        #     print(nii_path)
        #     break

        # # 将切片添加到已处理过的集合中
        # processed_slices.add(slice_3d.tobytes())
        crop_size = (256, 256)
        if slice_3d.shape[1] < crop_size[0] or slice_3d.shape[2] < crop_size[1]:
            pad_height = max(0, crop_size[0] - slice_3d.shape[1])
            pad_width = max(0, crop_size[1] - slice_3d.shape[2])
            slice_3d = np.pad(slice_3d , ((0,0),(0, pad_height), (0, pad_width)), mode='constant')

        # 计算裁剪的起始位置
        start_x = (slice_3d.shape[1] - crop_size[0]) // 2
        start_y = (slice_3d.shape[2] - crop_size[1]) // 2

        # 计算裁剪的结束位置
        end_x = start_x + crop_size[0]
        end_y = start_y + crop_size[1]


        slice_3d = slice_3d[:,start_x:end_x, start_y:end_y]
        print(slice_3d.shape)
        if slice_3d.shape[0]>25:
            slice_3d = slice_3d[0:25,:,:] 
        slice_img = sitk.GetImageFromArray(slice_3d)
        # 重新设置空间信息
        slice_img.SetSpacing((original_spacing[0], original_spacing[1], original_spacing[3]))
        slice_img.SetDirection((1.0, 0.0, 0.0, 
                         0.0, 1.0, 0.0, 
                         0.0, 0.0, -1.0))
        
        
        # 保存切片
        file_path_3d = os.path.join(save_path, f'slice_{i+1}.nii.gz')
        sitk.WriteImage(slice_img, file_path_3d)
        

#data_path = "./database"

data_path ="/data2/zsn/Dataset/ACDC/database"
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith("4d.nii.gz"):
            file_path = os.path.join(root, file)
            save_path = os.path.join(root, "slices")
            save_slices_from_nii(file_path, save_path)