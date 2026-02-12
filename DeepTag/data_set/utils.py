import torch
import argparse
import glob
import os
import re
import random
import pydicom
import numpy as np

np.bool = bool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image
from medpy import metric
import nibabel as nib
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter
import cv2
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist


def data_augment(image, shift=10.0, rotate=10.0, scale=0.1, intensity=0.1, flip=False):
    # Perform affine transformation on image and label, which are 4D tensors of dimension (N, C, X, Y).
    image2 = np.zeros(image.shape, dtype='float32')
    for i in range(image.shape[0]):
        # Random affine transformation using normal distributions
        shift_var = [np.clip(np.random.normal(), -3, 3) * shift, np.clip(np.random.normal(), -3, 3) * shift]
        rotate_var = np.clip(np.random.normal(), -3, 3) * rotate
        scale_var = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_var = 1 + np.clip(np.random.normal(), -0.5, 0) * intensity

        # Apply affine transformation (rotation + scale + shift) to training images
        row, col = image.shape[2:]
        M = cv2.getRotationMatrix2D((col / 2, row / 2), rotate_var, scale_var)
        M[0, 2] += shift_var[0]
        M[1, 2] += shift_var[1]

        for c in range(image.shape[1]):
            image2[i, c] = ndimage.affine_transform(image[i, c], M[:, :2], offset=M[:, 2], order=1)

        # Apply intensity variation
        if np.random.uniform() >= 0.67:
            image2[i, :] *= intensity_var

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.67:
                image2[i, :] = image2[i, :, ::-1, :]
            elif np.random.uniform() <= 0.33:
                image2[i, :] = image2[i, :, :, ::-1]

    return image2


def dicom_to_tensor(dicom_file_path, target_size=(288, 288)):
    ds = pydicom.dcmread(dicom_file_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # 归一化到 [0, 1]
    img = (img * 255).astype(np.uint8)  # 转换为 uint8
    img = Image.fromarray(img)
    img_resized = img.resize(target_size)

    return T.to_tensor(img_resized)


def sitk_to_tensor(mask_file_path, target_size=(192, 192)):
    img = sitk.ReadImage(mask_file_path)
    mask = sitk.GetArrayFromImage(img)
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))  # Normalize mask to [0, 1]
    mask = (mask * 255).astype(np.uint8)
    mask = resize(mask, target_size)
    mask = torch.tensor(mask)

    return mask


def normalize_data(img_np):
    # preprocessing
    cm = np.median(img_np)
    img_np = img_np / (8 * cm)
    img_np[img_np < 0] = 0.0
    img_np[img_np > 1.0] = 1.0
    return img_np


def load_nii_to_tensor(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    normed_data = normalize_data(data)
    return normed_data


def load_nii_to_tensor2(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normed_data


def load_nii_to_tensor255(file_path):
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data * 255).astype(np.uint8)
    tensor_data = torch.tensor(data)

    return tensor_data


def read_info_cfg(file_path):
    info = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if key in ['ED', 'ES', 'NbFrame']:
                info[key] = int(value)
            elif key in ['Height', 'Weight']:
                info[key] = float(value)
            else:
                info[key] = value
    return info


def get_es_value(file_path):
    config_path = os.path.dirname(os.path.dirname(file_path))
    config_path = os.path.join(config_path, 'Info.cfg')
    info = read_info_cfg(config_path)
    return info.get('ED'), info.get('ES'), info.get('Group')


def mask_tensor(file_path):
    ed_value, es_value, group = get_es_value(file_path)

    str_ed_value = "{:02d}".format(ed_value)
    ed_mask_path = file_path.replace('.nii.gz', '_frame' + str_ed_value + '.nii.gz')
    ed_mask = sitk.ReadImage(ed_mask_path)
    ed_mask = sitk.GetArrayFromImage(ed_mask)
    ed_mask = torch.tensor(ed_mask)
    ed_mask = ed_mask.unsqueeze(0)

    str_es_value = "{:02d}".format(es_value)
    es_mask_path = ed_mask_path.replace('_frame01.nii.gz', '_frame' + str_es_value + '.nii.gz')
    es_mask = sitk.ReadImage(es_mask_path)
    es_mask = sitk.GetArrayFromImage(es_mask)
    es_mask = torch.tensor(es_mask)
    es_mask = es_mask.unsqueeze(0)

    return ed_mask, es_mask, es_value, group


class TagDataset(Dataset):
    def __init__(self, data_path, data_type):
        super().__init__()
        self.data_path, self.data_type = data_path, data_type
        fp = open('{}/{}/cine_files.txt'.format(self.data_path, self.data_type), 'r')
        imgseqs = []
        for line in fp:
            line = line.strip('\n')
            line = line.strip()
            if line:
                imgseqs.append(line)
        self.imgseqs = imgseqs
        self.num = len(self.imgseqs)
        self.indices = list(range(self.num))
        if self.data_type == 'train':
            random.shuffle(self.indices)

    def __len__(self):
        return self.num

    def __getitem__(self, indx):
        idx = self.indices[indx % self.num]
        seq_path = self.imgseqs[idx]
        cine_imgs = load_nii_to_tensor(seq_path)

        ed_value, es_value, group = get_es_value(seq_path)
        if self.data_type == 'train':
            cine_imgs = cine_imgs + np.random.normal(scale=0.01, size=cine_imgs.shape)
            cine_imgs = data_augment(cine_imgs[np.newaxis], shift=10.0, rotate=10.0, scale=0.1,
                                     intensity=0.1, flip=True)
            cine_imgs = np.squeeze(cine_imgs)
            return seq_path, torch.tensor(cine_imgs), es_value
        elif self.data_type == 'test':
            ed_mask, es_mask, es_value, group = mask_tensor(seq_path)
            return seq_path, torch.tensor(cine_imgs), ed_mask, es_mask, es_value, group
        else:
            return seq_path, torch.tensor(cine_imgs), es_value



class TagDataset1(Dataset):
    def __init__(self, data_path, data_type):
        super().__init__()
        self.data_path, self.data_type = data_path, data_type
        fp = open('{}/cine_files.txt'.format(self.data_path, self.data_type), 'r')
        imgseqs = []
        for line in fp:
            line = line.strip('\n')
            line = line.strip()
            if line:
                imgseqs.append(line)
        self.imgseqs = imgseqs
        self.num = len(self.imgseqs)
        self.indices = list(range(self.num))
        if self.data_type == 'train':
            random.shuffle(self.indices)

    def __len__(self):
        return self.num

    def __getitem__(self, indx):
        idx = self.indices[indx % self.num]
        seq_path = self.imgseqs[idx]
        cine_imgs = load_nii_to_tensor(seq_path)

        if self.data_type == 'train':
            # 训练时需要ES值和数据增强
            ed_value, es_value, group = get_es_value(seq_path)
            cine_imgs = cine_imgs + np.random.normal(scale=0.01, size=cine_imgs.shape)
            cine_imgs = data_augment(cine_imgs[np.newaxis], shift=10.0, rotate=10.0, scale=0.1,
                                     intensity=0.1, flip=True)
            cine_imgs = np.squeeze(cine_imgs)
            return seq_path, torch.tensor(cine_imgs), es_value
        elif self.data_type == 'test':
            # 测试时只返回路径和图像数据，不需要mask和标签
            return seq_path, torch.tensor(cine_imgs)
        else:
            # validation模式
            ed_value, es_value, group = get_es_value(seq_path)
            return seq_path, torch.tensor(cine_imgs), es_value


class SynDataset(Dataset):
    def __init__(self, data_path, data_type, transform=None):
        self.data_path, self.data_type = data_path, data_type
        self.data_dir = os.path.join(self.data_path, self.data_type)
        self.transform = transform
        self.file_list = [f for f in os.listdir(self.data_dir) if not f.endswith('.txt')]

    def __len__(self):
        # 返回数据集的大小
        return len(self.file_list)

    def __getitem__(self, idx):
        # 获取第 idx 个样本
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        file_list = [f for f in os.listdir(file_path) if 'frame' in f]
        file_list = sorted(file_list, key=sort_key)
        ed_path, ed_mask_path, es_path, es_mask_path = [os.path.join(file_path, f) for f in file_list]

        return ed_path, ed_mask_path, es_path, es_mask_path


def sort_key(filename):
    match = re.search(r"frame(\d+)", filename)
    if match:
        frame_num = int(match.group(1))
        return (frame_num, '_gt' in filename)  # 按frame编号排序，并使_gt文件排在后面
    return (float('inf'), filename)  # 没有匹配的文件排在最后


def psnr(x, y, data_range=1.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=1, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def hausdorff_95(array1, array2, labels=None, include_zero=False):
    """
    Computes the 95th percentile Hausdorff distance (HD95) between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute HD95 on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.

    Returns:
        Array of HD95 distances for each label and the average.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    hd95_distances = np.zeros(len(labels))

    for b in range(array1.shape[0]):
        array1_2d = array1[b, 0, ::]
        array2_2d = array2[b, 0, ::]

        for idx, label in enumerate(labels):
            points1 = np.argwhere(array1_2d == label)
            points2 = np.argwhere(array2_2d == label)

            if points1.size == 0 or points2.size == 0:
                hd95_distances[idx] = np.nan  # Handle cases where there are no points for the label
            else:
                # Calculate directed Hausdorff distances in both directions
                distances_AB = [directed_hausdorff(points1, points2)[0]]
                distances_BA = [directed_hausdorff(points2, points1)[0]]

                # Combine the distances
                all_distances = np.concatenate([distances_AB, distances_BA])

                # Compute the 95th percentile of distances
                hd95_distances[idx] = np.percentile(all_distances, 95)

    # Calculate the average HD95 across all labels, ignoring NaN values
    avg_hd95 = np.nanmean(hd95_distances)
    return np.append(hd95_distances, avg_hd95)


def average_hausdorff(array1, array2, labels=None, include_zero=False):
    """
    Computes the Average Hausdorff distance between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute Hausdorff distance on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.

    Returns:
        Array of Average Hausdorff distances for each label and the overall average.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    avg_hausdorff_distances = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        points1 = np.argwhere(array1 == label)
        points2 = np.argwhere(array2 == label)

        if points1.size == 0 or points2.size == 0:
            avg_hausdorff_distances[idx] = np.nan  # Handle cases where there are no points for the label
        else:
            # Compute the pairwise distances between points in the two sets
            d_matrix = cdist(points1, points2, 'euclidean')

            # For each point in array1, find the closest point in array2 and vice versa, then average
            min_d_AB = np.mean(np.min(d_matrix, axis=1))
            min_d_BA = np.mean(np.min(d_matrix, axis=0))

            # The average Hausdorff distance is the average of these two minimum distances
            avg_hausdorff_distances[idx] = (min_d_AB + min_d_BA) / 2

    overall_avg_hd = np.nanmean(avg_hausdorff_distances)  # Calculate the overall average, ignoring NaN values
    return np.append(avg_hausdorff_distances, overall_avg_hd)


def compute_metrics(segmentation, prediction, label):
    """
    计算特定标签的95% Hausdorff距离(HD95)和dice

    参数:
        segmentation (numpy.ndarray): 分割标签图像
        prediction (numpy.ndarray): 分割预测图像
        label (int): 要计算的特定类别标签

    返回:
        float: 95% Hausdorff距离和dice
    """
    binary_gt = (segmentation == label).astype(np.uint8)
    binary_pred = (prediction > 0).astype(np.uint8)
    try:
        hd95 = metric.binary.hd95(binary_gt, binary_pred)
        dice = metric.binary.dc(binary_pred, binary_gt)
    except RuntimeError:
        return np.inf, np.inf  # 如果没有边缘点，则返回无穷大
    return hd95, dice


def compute_metrics_all_classes(prediction, segmentation):
    """
    计算所有类别的95% Hausdorff距离(HD95)和dice

    参数:
        segmentation (numpy.ndarray): 分割标签图像
        prediction (numpy.ndarray): 分割预测图像

    返回:
        dict: 每个类别的95% Hausdorff距离 和dice
    """
    labels = np.unique(segmentation[segmentation > 0])  # 获取所有前景类别（假设0是背景）
    hd95_dict = {}
    dice_dict = {}
    for label in labels:
        hd95, dice = compute_metrics(segmentation, prediction[int(label) - 1], label)
        hd95_dict[label] = hd95
        dice_dict[label] = dice

    return hd95_dict, dice_dict


# def extract_contour_points(mask):
#     """
#     Extract the contour points from a binary mask.

#     Parameters:
#     - mask: A binary mask of shape (160, 160).

#     Returns:
#     - points: A set of points representing the contour.
#     """
#     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     points = np.vstack(contours).squeeze(1)
#     return points

# def hausdorff_distance(set1, set2):
#     """
#     Calculate the Hausdorff Distance between two sets of points.

#     Parameters:
#     - set1: First set of points as a numpy array of shape (n_points1, 2).
#     - set2: Second set of points as a numpy array of shape (n_points2, 2).

#     Returns:
#     - hd: Hausdorff Distance between the two sets of points.
#     """
#     d1 = directed_hausdorff(set1, set2)[0]
#     d2 = directed_hausdorff(set2, set1)[0]
#     hd = max(d1, d2)
#     return hd

# def hausdorff(masks1, masks2, num_classes=3):
#     """
#     Calculate the Hausdorff Distance for each class in a batch of multi-class masks.

#     Parameters:
#     - masks1: Batch of multi-class masks with shape (b, 1, 160, 160).
#     - masks2: Batch of multi-class masks with shape (b, 1, 160, 160).
#     - num_classes: Number of classes (excluding background).

#     Returns:
#     - hd_dict: A dictionary with class indices as keys and Hausdorff Distances as values.
#     """
#     batch_size = masks1.shape[0]
#     hd_dict = {i: [] for i in range(1, num_classes + 1)}

#     for i in range(batch_size):
#         for cls in range(1, num_classes + 1):
#             set1 = extract_contour_points((masks1[i, 0] == cls).astype(np.uint8))
#             set2 = extract_contour_points((masks2[i, 0] == cls).astype(np.uint8))

#             if set1.size == 0 or set2.size == 0:
#                 hd = np.inf  # Handle the case where contours are not present
#             else:
#                 hd = hausdorff_distance(set1, set2)

#             hd_dict[cls].append(hd)

#     return hd_dict

def plt_points(points1, points2):
    """
    Visualize two sets of points on a scatter plot.

    Parameters:
    - points1: First set of points as a numpy array of shape (n_points1, 2).
    - points2: Second set of points as a numpy array of shape (n_points2, 2).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(points1[:, 0], points1[:, 1], c='blue', label='Points1')
    plt.scatter(points2[:, 0], points2[:, 1], c='red', label='Points2')
    plt.legend()
    plt.title("Visualization of Two Point Sets")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig('./points.png')


