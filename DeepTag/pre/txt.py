#(5)

import os
import re
# 定义数据文件夹路径
#data_folder = "./database/train/"

data_folder ="/code/zsn/Dataset/cine_lge/cine-mri"

output_file = "/code/zsn/Dataset/cine_lge/cine-mri/cine_files.txt"
def natural_sort_key(s):
    """
       该函数用于生成一个可作为排序依据的键列表，以实现自然排序的效果。

       自然排序即按照人类习惯的方式对包含数字和字母的字符串进行排序，
       例如，使得 "file1.txt"、"file2.txt"、"file10.txt" 能按照数字的实际大小顺序排序，
       而不是按照字符串的字典序（默认排序方式下 "file1.txt"、"file10.txt"、"file2.txt" 的顺序是错误的）。

       参数：
       s (str)：需要生成排序键的输入字符串。

       返回值：
       list：一个经过处理的列表，其中数字部分被转换为整数，字母部分被转换为小写，
             该列表将作为排序的键，使得排序按照自然顺序进行。
       """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
"""
# with open(output_file, "w") as f:
#     # 遍历所有病人文件夹
#     for number_folder in os.listdir(data_folder):
#         number_folder = os.path.join(data_folder, number_folder)
# 
#         for date_folder in os.listdir(number_folder):
#             if "slices" in date_folder:
#                 date_folder = os.path.join(number_folder, date_folder)
# 
#                 for cine_folder in  sorted(os.listdir(date_folder), key=natural_sort_key):
#                     cine_path = os.path.join(date_folder, cine_folder)
#                     f.write(cine_path + "\n")
"""

with open(output_file, "w") as f:
    # 遍历所有病人文件夹
    for number_folder in os.listdir(data_folder):  # 遍历 data_folder 中的所有项目
        number_folder = os.path.join(data_folder, number_folder)  # 构造病人文件夹路径

        if not os.path.isdir(number_folder):  # 检查是否是文件夹，跳过文件
            continue

        # 遍历病人文件夹中的所有日期文件夹
        for date_folder in os.listdir(number_folder):  # 遍历子文件夹
            if "slices" in date_folder:  # 筛选文件夹名中包含 "slices" 的文件夹
                date_folder = os.path.join(number_folder, date_folder)  # 构造日期文件夹路径

                if not os.path.isdir(date_folder):  # 再次检查是否为文件夹
                    continue

                # 遍历 "slices" 文件夹中的所有 cine 文件夹，并按自然顺序排序
                for cine_folder in sorted(os.listdir(date_folder), key=natural_sort_key):
                    cine_path = os.path.join(date_folder, cine_folder)  # 构造 cine 文件夹路径
                    f.write(cine_path + "\n")  # 将路径写入文件

""" # 创建一个文本文件来存储文件路径
output_file = "cine_files.txt"
with open(output_file, "w") as f:
    # 遍历所有日期文件夹
    for number_folder in os.listdir(data_folder):
        number_folder = os.path.join(data_folder, number_folder)
        for date_folder in os.listdir(number_folder):
            date_folder = os.path.join(number_folder, date_folder)
            for cine_folder in os.listdir(date_folder):
                
                if "CINE-2CH"  in cine_folder:
                    cine_path = os.path.join(date_folder, cine_folder)
                    for root, _, files in os.walk(cine_path):
                        files.sort()
                    for i, file_name in enumerate(files):
                        file_path = os.path.join(root, file_name)
                        # 检查是否是每行的最后一个路径
                        if i == len(files) - 1:
                            f.write(file_path)
                        else:
                            f.write(file_path + ",")
                    f.write("\n")
                
                if "CINE-3CH"  in cine_folder:
                    cine_path = os.path.join(date_folder, cine_folder)
                    for root, _, files in os.walk(cine_path):
                        files.sort()
                    for i, file_name in enumerate(files):
                        file_path = os.path.join(root, file_name)
                        # 检查是否是每行的最后一个路径
                        if i == len(files) - 1:
                            f.write(file_path)
                        else:
                            f.write(file_path + ",")
                    f.write("\n")

                if "CINE-4CH"  in cine_folder:
                    cine_path = os.path.join(date_folder, cine_folder)
                    for root, _, files in os.walk(cine_path):
                        files.sort()
                    for i, file_name in enumerate(files):
                        file_path = os.path.join(root, file_name)
                        # 检查是否是每行的最后一个路径
                        if i == len(files) - 1:
                            f.write(file_path)
                        else:
                            f.write(file_path + ",")
                    f.write("\n")
                
                if "CINE-SA"  in cine_folder:
                    cine_path = os.path.join(date_folder, cine_folder)
                    for root, _, files in os.walk(cine_path):
                        files.sort()
                    for i, file_name in enumerate(files):
                        file_path = os.path.join(root, file_name)
                        # 检查是否是每行的最后一个路径
                        if i == len(files) - 1:
                            f.write(file_path)
                        else:
                            f.write(file_path + ",")
                    f.write("\n")

print("文件路径已写入到", output_file) """




