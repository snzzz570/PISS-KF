##（2）

import SimpleITK as sitk
import os

def reorient_to_rai(image):
    # 获取图像的方向、原点和间距
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    
    # 将图像方向重新调整为 LPS

    if image.GetDimension() == 3:
        new_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    elif image.GetDimension() == 4:
        new_direction = (1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0)
    elif image.GetDimension() == 2:
        new_direction = (1.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError("Unsupported image dimension: {}".format(image.GetDimension()))
    
    # 创建新图像并设置方向、原点和间距
    lps_image = sitk.Image(image)
    lps_image.SetDirection(new_direction)
    lps_image.SetOrigin(origin)
    lps_image.SetSpacing(spacing)
    
    return lps_image

def reorient_to_ras(image):
    # 获取图像的方向、原点和间距
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()

    if image.GetDimension() == 2:
        # 2D图像的方向矩阵
        new_direction = (1.0, 0.0, 0.0, -1.0)
    elif image.GetDimension() == 3:
        # 3D图像的方向矩阵
        new_direction = (1.0, 0.0, 0.0, 
                         0.0, 1.0, 0.0, 
                         0.0, 0.0, -1.0)
    elif image.GetDimension() == 4:
        # 4D图像的方向矩阵
        new_direction = (1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, -1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0)
    else:
        raise ValueError("Unsupported image dimension: {}".format(image.GetDimension()))

    # 创建新图像并设置方向、原点和间距
    ras_image = sitk.Image(image)
    ras_image.SetDirection(new_direction)
    ras_image.SetOrigin(origin)
    ras_image.SetSpacing(spacing)

    return ras_image

def reorient_to_lps(image):
    dimension = image.GetDimension()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()

    if dimension == 2:
        # 2D图像的方向矩阵
        new_direction = (-1.0, 0.0, 0.0, -1.0)
        new_origin = (-origin[0], -origin[1])
    elif dimension == 3:
        # 3D图像的方向矩阵
        new_direction = (-direction[0], -direction[1], -direction[2], 
                         -direction[3], -direction[4], -direction[5], 
                         direction[6], direction[7], direction[8])
        new_origin = (-origin[0], -origin[1], origin[2])
    elif dimension == 4:
        # 4D图像的方向矩阵
        new_direction = (-direction[0], -direction[1], -direction[2], -direction[3],
                         -direction[4], -direction[5], -direction[6], -direction[7],
                         -direction[8], -direction[9], direction[10], direction[11],
                         direction[12], direction[13], direction[14], direction[15])
        new_origin = (-origin[0], -origin[1], origin[2], origin[3])
    else:
        raise ValueError("Unsupported image dimension: {}".format(dimension))

    # 创建新图像并设置方向、原点和间距
    lps_image = sitk.Image(image)
    lps_image.SetDirection(new_direction)
    lps_image.SetOrigin(new_origin)
    lps_image.SetSpacing(spacing)

    return lps_image
def reorient_to_lpi(image):
    dimension = image.GetDimension()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    spacing = image.GetSpacing()

    if dimension == 2:
        # 2D图像的方向矩阵
        new_direction = (-1.0, 0.0, 0.0, -1.0)
        new_origin = (-origin[0], -origin[1])
    elif dimension == 3:
        # 3D图像的方向矩阵
        new_direction = (-direction[0], -direction[1], direction[2], 
                         -direction[3], -direction[4], direction[5], 
                         -direction[6], -direction[7], direction[8])
        new_origin = (-origin[0], -origin[1], -origin[2])
    elif dimension == 4:
        # 4D图像的方向矩阵
        new_direction = (-direction[0], -direction[1], direction[2], direction[3],
                         -direction[4], -direction[5], direction[6], direction[7],
                         -direction[8], -direction[9], -direction[10], direction[11],
                         -direction[12], -direction[13], direction[14], direction[15])
        new_origin = (-origin[0], -origin[1], -origin[2], origin[3])
    else:
        raise ValueError("Unsupported image dimension: {}".format(dimension))

    # 创建新图像并设置方向、原点和间距
    lpi_image = sitk.Image(image)
    lpi_image.SetDirection(new_direction)
    lpi_image.SetOrigin(new_origin)
    lpi_image.SetSpacing(spacing)

    return lpi_image
# def reorient_to_lps(image):
#     dimension = image.GetDimension()
#     direction = image.GetDirection()
#     origin = image.GetOrigin()
#     spacing = image.GetSpacing()

#     if dimension == 2:
#         # 2D图像的方向矩阵
#         new_direction = (direction[0], -direction[1], 
#                          -direction[2], direction[3])
#     elif dimension == 3:
#         # 3D图像的方向矩阵
#         new_direction = (-direction[0], -direction[1], direction[2], 
#                          -direction[3], -direction[4], direction[5], 
#                          -direction[6], -direction[7], direction[8])
#     elif dimension == 4:
#         # 4D图像的方向矩阵
#         new_direction = (-direction[0], -direction[1], -direction[2], direction[3],
#                          -direction[4], -direction[5], -direction[6], direction[7],
#                          -direction[8], -direction[9], -direction[10], direction[11],
#                          -direction[12], -direction[13], -direction[14], direction[15])
#     else:
#         raise ValueError("Unsupported image dimension: {}".format(dimension))
    
#     # 创建新图像并设置方向、原点和间距
#     lps_image = sitk.Image(image)
#     lps_image.SetDirection(new_direction)
#     lps_image.SetOrigin(origin)
#     lps_image.SetSpacing(spacing)
    
#     return lps_image
# # 读取图像
data_path = "/data2/zsn/Dataset/ACDC/database"
for root,dirs,files in os.walk(data_path):
    for file in files:
        if  file.endswith('.nii.gz'):
            file_path = os.path.join(root, file)
            print(file_path)
            file_img = sitk.ReadImage(file_path)
            # 转换坐标系为LPS
            lps_image = reorient_to_ras(file_img)

            sitk.WriteImage(lps_image, file_path)