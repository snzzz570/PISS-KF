import torch
import os
import random
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from scipy import ndimage
import cv2


def data_augment(image, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=False):
    """
    对图像进行数据增强，包括平移、旋转、缩放、强度变化和翻转

    参数:
        image: 4D张量，维度为(N, C, X, Y)
        shift: 平移范围
        rotate: 旋转角度范围
        scale: 缩放比例范围
        intensity: 强度变化范围
        flip: 是否进行翻转

    返回:
        增强后的图像
    """
    image2 = np.zeros(image.shape, dtype='float32')

    for i in range(image.shape[0]):
        # 随机仿射变换参数，使用正态分布
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift,
                     np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -0.5, 0) * intensity

        # 应用仿射变换（旋转 + 缩放 + 平移）
        row, col = image.shape[2:]
        M = cv2.getRotationMatrix2D((col / 2, row / 2), rotate_var, scale_var)
        M[0, 2] += shift_var[0]
        M[1, 2] += shift_var[1]

        for c in range(image.shape[1]):
            image2[i, c] = ndimage.affine_transform(image[i, c], M[:, :2],
                                                    offset=M[:, 2], order=1)

        # 应用强度变化
        if np.random.uniform() >= 0.67:
            image2[i, :] *= intensity_var

        # 应用随机水平或垂直翻转
        if flip:
            if np.random.uniform() >= 0.67:
                image2[i, :] = image2[i, :, ::-1, :]  # 水平翻转
            elif np.random.uniform() <= 0.33:
                image2[i, :] = image2[i, :, :, ::-1]  # 垂直翻转

    return image2


def normalize_data(img_np):
    """
    数据标准化处理

    参数:
        img_np: 输入图像numpy数组

    返回:
        标准化后的图像
    """
    cm = np.median(img_np)
    img_np = img_np / (8 * cm)
    img_np[img_np < 0] = 0.0
    img_np[img_np > 1.0] = 1.0
    return img_np


def load_nii_to_tensor(file_path):
    """
    加载NII文件并转换为张量

    参数:
        file_path: NII文件路径

    返回:
        标准化后的numpy数组
    """
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    normed_data = normalize_data(data)
    return normed_data


class SimplifiedTrainDataset(Dataset):
    """
    简化的训练数据集类
    去除了配置文件相关的功能，直接从文件路径加载数据
    """

    def __init__(self, data_path, data_type='train', file_list_name='cine_files.txt'):
        """
        初始化数据集

        参数:
            data_path: 数据根目录路径
            data_type: 数据类型 ('train', 'val', 'test')
            file_list_name: 文件列表名称
        """
        super().__init__()
        self.data_path = data_path
        self.data_type = data_type

        # 如果没有文件列表，则自动扫描目录中的.nii.gz文件
        file_list_path = os.path.join(self.data_path, self.data_type, file_list_name)
        self.imgseqs = []

        if os.path.exists(file_list_path):
            # 从文件列表读取
            with open(file_list_path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        self.imgseqs.append(line)
        else:
            # 自动扫描目录
            data_dir = os.path.join(self.data_path, self.data_type)
            for file in os.listdir(data_dir):
                if file.endswith('.nii.gz') or file.endswith('.nii'):
                    self.imgseqs.append(os.path.join(data_dir, file))

        self.num = len(self.imgseqs)
        if self.num == 0:
            raise ValueError(f"在路径 {os.path.join(self.data_path, self.data_type)} 中没有找到任何数据文件")

        self.indices = list(range(self.num))

        # 如果是训练模式，打乱数据顺序
        if self.data_type == 'train':
            random.shuffle(self.indices)

    def __len__(self):
        """返回数据集大小"""
        return self.num

    def __getitem__(self, indx):
        """
        获取数据项

        参数:
            indx: 数据索引

        返回:
            训练模式: (文件路径, 图像张量)
            测试模式: (文件路径, 图像张量)
        """
        try:
            idx = self.indices[indx % self.num]
            seq_path = self.imgseqs[idx]

            # 加载图像数据
            cine_imgs = load_nii_to_tensor(seq_path)

            # 确保数据是浮点型
            cine_imgs = cine_imgs.astype(np.float32)

            if self.data_type == 'train':
                # 训练模式：添加噪声和数据增强
                # 添加高斯噪声
                noise = np.random.normal(scale=0.01, size=cine_imgs.shape)
                cine_imgs = cine_imgs + noise

                # 数据增强 - 确保输入是4D
                if len(cine_imgs.shape) == 3:
                    # (D, H, W) -> (1, D, H, W)
                    cine_imgs_4d = cine_imgs[np.newaxis, :, :, :]
                elif len(cine_imgs.shape) == 4:
                    # 已经是4D
                    cine_imgs_4d = cine_imgs
                else:
                    raise ValueError(f"不支持的图像维度: {cine_imgs.shape}")

                # 应用数据增强
                try:
                    augmented_imgs = data_augment(cine_imgs_4d,
                                                  shift=10.0,
                                                  rotate=10.0,
                                                  scale=0.1,
                                                  intensity=0.1,
                                                  flip=True)
                    cine_imgs = np.squeeze(augmented_imgs)
                except Exception as e:
                    print(f"数据增强失败，使用原始数据: {e}")
                    cine_imgs = np.squeeze(cine_imgs_4d)

            # 确保张量形状一致性
            if len(cine_imgs.shape) == 2:
                # (H, W) -> (1, H, W)
                cine_imgs = cine_imgs[np.newaxis, :, :]
            elif len(cine_imgs.shape) == 3:
                # (D, H, W) 保持不变
                pass
            elif len(cine_imgs.shape) == 4:
                # (B, D, H, W) -> (D, H, W)
                cine_imgs = np.squeeze(cine_imgs)

            # 转换为torch张量
            tensor_imgs = torch.tensor(cine_imgs, dtype=torch.float32)

            return seq_path, tensor_imgs

        except Exception as e:
            print(f"加载数据时出错 {seq_path}: {e}")
            # 返回一个默认的张量以避免程序崩溃
            dummy_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
            return seq_path, dummy_tensor


# class SimplifiedMaskDataset(Dataset):
#     """
#     简化的掩码数据集类
#     用于处理带有掩码标签的数据
#     """
#
#     def __init__(self, data_path, data_type='train', image_dir='images', mask_dir='masks'):
#         """
#         初始化掩码数据集
#
#         参数:
#             data_path: 数据根目录路径
#             data_type: 数据类型 ('train', 'val', 'test')
#             image_dir: 图像文件夹名称
#             mask_dir: 掩码文件夹名称
#         """
#         super().__init__()
#         self.data_path = data_path
#         self.data_type = data_type
#         self.image_dir = os.path.join(data_path, data_type, image_dir)
#         self.mask_dir = os.path.join(data_path, data_type, mask_dir)
#
#         # 获取图像文件列表
#         self.image_files = [f for f in os.listdir(self.image_dir)
#                             if f.endswith('.nii.gz') or f.endswith('.nii')]
#         self.image_files.sort()
#
#         self.num = len(self.image_files)
#         self.indices = list(range(self.num))
#
#         if self.data_type == 'train':
#             random.shuffle(self.indices)
#
#     def __len__(self):
#         """返回数据集大小"""
#         return self.num
#
#     def __getitem__(self, indx):
#         """
#         获取数据项
#
#         参数:
#             indx: 数据索引
#
#         返回:
#             (图像路径, 图像张量, 掩码路径, 掩码张量)
#         """
#         idx = self.indices[indx % self.num]
#         image_file = self.image_files[idx]
#
#         # 构建完整路径
#         image_path = os.path.join(self.image_dir, image_file)
#         mask_path = os.path.join(self.mask_dir, image_file)
#
#         # 加载图像和掩码
#         image_data = load_nii_to_tensor(image_path)
#         mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
#
#         if self.data_type == 'train':
#             # 训练模式：同时对图像和掩码进行相同的几何变换
#             # 添加噪声（仅对图像）
#             image_data = image_data + np.random.normal(scale=0.01, size=image_data.shape)
#
#             # 将图像和掩码合并进行相同的几何变换
#             combined_data = np.concatenate([image_data[np.newaxis],
#                                             mask_data[np.newaxis]], axis=1)
#
#             # 应用数据增强
#             augmented_data = data_augment(combined_data,
#                                           shift=10.0,
#                                           rotate=10.0,
#                                           scale=0.1,
#                                           intensity=0.0,  # 掩码不应用强度变化
#                                           flip=True)
#
#             # 分离图像和掩码
#             image_data = np.squeeze(augmented_data[:, :image_data.shape[0]])
#             mask_data = np.squeeze(augmented_data[:, image_data.shape[0]:])
#
#         return image_path, torch.tensor(image_data), mask_path, torch.tensor(mask_data)


# 使用示例
def create_data_loader(data_path, data_type='train', batch_size=4, shuffle=True, num_workers=0):
    """
    创建数据加载器

    参数:
        data_path: 数据路径
        data_type: 数据类型
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作线程数 (Windows下建议设为0)

    返回:
        DataLoader对象
    """
    dataset = SimplifiedTrainDataset(data_path, data_type)

    from torch.utils.data import DataLoader

    def collate_fn(batch):
        """自定义批处理函数，处理不同尺寸的张量"""
        paths, tensors = zip(*batch)

        # 找到最大的尺寸
        max_dims = []
        for i in range(len(tensors[0].shape)):
            max_dim = max(tensor.shape[i] for tensor in tensors)
            max_dims.append(max_dim)

        # 将所有张量填充到相同尺寸
        padded_tensors = []
        for tensor in tensors:
            # 计算需要填充的大小
            padding = []
            for i in range(len(tensor.shape)):
                pad_size = max_dims[i] - tensor.shape[i]
                padding.extend([0, pad_size])

            # 反转padding顺序（PyTorch的pad函数要求从最后一个维度开始）
            padding = padding[::-1]

            # 进行填充
            padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
            padded_tensors.append(padded_tensor)

        # 堆叠张量
        stacked_tensors = torch.stack(padded_tensors, dim=0)

        return list(paths), stacked_tensors

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=collate_fn)

    return dataloader


# 使用示例代码
if __name__ == "__main__":
    # 设置数据路径 - 根据您的实际路径
    data_path = r"/data/zsn/Data/database11"  # 您的数据根目录

    # 创建训练数据加载器
    train_loader = create_data_loader(data_path, 'train', batch_size=4)

    # 创建验证数据加载器
    val_loader = create_data_loader(data_path, 'val', batch_size=4, shuffle=False)

    print("数据集信息:")
    print(f"训练数据路径: {os.path.join(data_path, 'train')}")
    print(f"验证数据路径: {os.path.join(data_path, 'val')}")

    # 测试训练数据加载
    print("\n=== 训练数据测试 ===")
    for i, (seq_paths, cine_imgs) in enumerate(train_loader):
        print(f"批次 {i + 1}:")
        print(f"  序列路径: {seq_paths}")
        print(f"  图像形状: {cine_imgs.shape}")
        print(f"  图像数据类型: {cine_imgs.dtype}")

        if i >= 2:  # 只显示前3个批次
            break

    # 测试验证数据加载
    print("\n=== 验证数据测试 ===")
    for i, (seq_paths, cine_imgs) in enumerate(val_loader):
        print(f"批次 {i + 1}:")
        print(f"  序列路径: {seq_paths}")
        print(f"  图像形状: {cine_imgs.shape}")
        print(f"  图像数据类型: {cine_imgs.dtype}")

        if i >= 1:  # 只显示前2个批次
            break