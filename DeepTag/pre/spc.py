import torch
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
#data_path = './database/train'
data_path ="E:\\PythonProject\\database\\train"

for root,dirs,files in os.walk(data_path):
    for file in files:
        if  file.endswith('.nii.gz'):
            file_path = os.path.join(root, file)
            print(file_path)
            file_img = sitk.ReadImage(file_path)
            spacing1 = file_img.GetSpacing()
            print(spacing1)


def resample_image(image, new_spacing=[1.5, 1.5], new_size=[128, 128]):
    # 获取原始图像的尺寸和间距
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # 计算新尺寸
    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(len(original_size))
    ]

    # 创建重新采样器
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    # 重新采样图像
    resampled_image = resampler.Execute(image)

    # 中心裁剪到128x128
    original_array = sitk.GetArrayFromImage(resampled_image)
    z, y, x = original_array.shape
    startx = x//2 - (new_size[0]//2)
    starty = y//2 - (new_size[1]//2)
    cropped_array = original_array[:, starty:starty+new_size[1], startx:startx+new_size[0]]

    cropped_image = sitk.GetImageFromArray(cropped_array)
    cropped_image.SetSpacing(resampled_image.GetSpacing())
    cropped_image.SetOrigin(resampled_image.GetOrigin())
    cropped_image.SetDirection(resampled_image.GetDirection())

    return cropped_image

# 读取图像
# image_path = "path_to_your_image.nii"  # 请替换为实际图像路径
# image = sitk.ReadImage(image_path)

# # 重新采样并裁剪图像
# processed_image = resample_image(image)

# # 保存处理后的图像
# output_path = "path_to_save_processed_image.nii"  # 请替换为实际保存路径
# sitk.WriteImage(processed_image, output_path)