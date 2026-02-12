#(3)

import os
import SimpleITK as sitk
import configparser
import random
import shutil

def read_cfg_simple(cfg_path):
    config = {}
    with open(cfg_path, 'r') as file:
        for line in file:
            if ': ' in line:
                key, value = line.strip().split(': ')
                config[key] = value
    return config['Group']

# dataroot = "./database/training"
# val_root = "./database/val"

dataroot = "/data2/zsn/Dataset/ACDC/database/training"
val_root ="/data2/zsn/Dataset/ACDC/database/test"
groups = {"NOR": [], "MINF": [], "DCM": [], "HCM": [], "RV": []}
for root, dirs, files in os.walk(dataroot):
    for file in files:
        if file.endswith(".cfg"):
            file_path = os.path.join(root, file)
            value = read_cfg_simple(file_path)
            if value in groups:
                groups[value].append(root)

for group, paths in groups.items():
    print(f"{group}: {len(paths)}")

# 为每个组随机选择4个路径并移动到val文件夹
for group, paths in groups.items():
    if len(paths) >= 4:  # 确保有足够的路径可供选择
        selected_paths = random.sample(paths, 4)  # 随机选择4个路径
        for path in selected_paths:
            destination = os.path.join(val_root, os.path.basename(path))
            if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination))  # 如果目标文件夹不存在，创建它
            shutil.move(path, destination)  # 移动文件夹
            print(f"Moved {path} to {destination}")
    else:
        print(f"Not enough paths in group {group} to select 4.")