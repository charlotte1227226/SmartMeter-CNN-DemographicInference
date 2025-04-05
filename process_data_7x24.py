import numpy as np
import os
import errno
import sys
from tqdm import tqdm
import time
import pandas as pd

# 刷新stdout缓冲区，以便及时输出信息
sys.stdout.flush()

# 定义输入文件路径列表
input_paths = [
    "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File1.txt",
    "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File2.txt",
    "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File3.txt",
    "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File4.txt",
    "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File5.txt",
    "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File6.txt"
]

# 定义输出文件路径
output_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_data_7x24/processed_data_7x24_6103_processed.npz"

# 创建空列表存储数据
train_images = []
train_labels = []

# 定义函数来处理每个输入文件
def process_input_file(input_path):
    start_time = time.time()

    # 读取输入文件的每一行，并将数据存储在一个列表中
    with open(input_path, "r") as f:
        lines = f.readlines()

    end_time1 = time.time()
    print(f"Time elapsed for current line: {end_time1 - start_time:.2f} seconds\n")

    data = []
    for line in tqdm(lines):
        line = line.strip().split()
        data.append([float(line[0]), float(line[1]), float(line[2])])

    end_time2 = time.time()

    print(f"Time elapsed for data reading: {end_time2 - end_time1:.2f} seconds\n")

    data = np.array(data)

    # 将数据转换为 Pandas DataFrame 格式
    df = pd.DataFrame(data, columns=['col1', 'col2', 'col3'])

    # 对数据进行处理
    df = df.groupby('col1').apply(lambda x: x.sort_values('col2')).reset_index(drop=True)
    df = df.sort_values(by=['col1', 'col2'])
    df = df.reset_index(drop=True)
    col1 = df['col1']

    # 读取Excel数据
    excel_data = pd.read_excel("/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/標籤檔/Q6103_processed.xlsx")  # 根据实际情况修改文件名和路径
    excel_col1 = excel_data.iloc[:, 0]
    excel_col2 = excel_data.iloc[:, 1]

    df = df[df['col1'].isin(excel_col1)]  # 相同ID保留 不同刪除

    df = df.groupby('col1').apply(lambda x: x.sort_values('col2')).reset_index(drop=True)
    df = df.sort_values(by=['col1', 'col2'])
    df = df.reset_index(drop=True)
    col1 = df['col1']

    # 提取col3并重塑数据
    col3_values = df['col3'].values
    label = []
    resized_data = []
    for i in tqdm(range(0, len(col3_values), 168), desc="Reshaping data"):
        for k in range(len(excel_col1)):
            if col1[i] == excel_col1[k]:
                chunk = col3_values[i:i+168].reshape((7, 24, 1))
                resized_data.append(chunk)
                label.append(excel_col2[k])
                break  # 找到匹配的值后，跳出内层循环

    end_time3 = time.time()
    print(f"Time elapsed for reshaping and saving data: {end_time3 - end_time2:.2f} seconds\n")
    
    # 将数据添加到训练集列表中
    train_images.extend(resized_data)
    train_labels.extend(label)

# 处理每个输入文件
for input_path in input_paths:
    process_input_file(input_path)

# 转换为NumPy数组
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# 保存数据集为NPZ文件
np.savez(output_path, images=train_images, labels=train_labels)
