# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
import errno
import sys
from tqdm import tqdm
import time
import pandas as pd
# 刷新stdout缓冲区，以便及时输出信息
sys.stdout.flush()

# 指定输入和输出文件路径
input_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/data/File6.txt"
output_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_data01~48/processed_File6.txt"

# 检查输出文件所在的目录是否存在，如果不存在，就创建它
if not os.path.exists(os.path.dirname(output_path)):
    try:
        os.makedirs(os.path.dirname(output_path))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
start_time = time.time()
# 读取输入文件的每一行，并将数据存储在一个列表中
with open(input_path, "r") as f:
    lines = f.readlines()
end_time1 = time.time()
print(f"Time elapsed for current line: {end_time1 - start_time:.2f} seconds\n")
data = []
# 将数据分割成三个部分，并将其存储在一个列表中
for line in tqdm(lines):
    line = line.strip().split()
    data.append([int(line[0]), int(line[1]), float(line[2])])
    # 记录时间
end_time2 = time.time()
# 将数据转换为numpy数组
data = np.array(data)
# 将数据转换为 Pandas DataFrame 格式
df = pd.DataFrame(data, columns=['col1', 'col2', 'col3'])
df = df.groupby('col1').apply(lambda x: x.sort_values('col2')).reset_index(drop=True)
print("\n1\n")
# 提取第二列并计算余数
col2 = df['col2'] % 100
print("2\n")
end_time2 = time.time()
print(f"Time elapsed for current line: {end_time2 - end_time1:.2f} seconds\n")
positions = []
for i in tqdm(range(len(col2) - 47), desc="Finding positions"):
    if col2[i] == 1:
        if all(col2[i+j] == j+1 for j in range(1, 48)):
            positions.append(i)
print("3\n")
end_time3 = time.time()
print(f"Time elapsed for current line: {end_time3 - end_time2:.2f} seconds\n")
if not positions:
    # 如果没有连续的01到48，则删除数据
    df = pd.DataFrame()
else:
    # 提取所有连续的01到48的数据
    new_data = []
    for start_pos in tqdm(positions, desc="Extracting continuous data"):
        new_data.append(df.iloc[start_pos:start_pos+48])
    df = pd.concat(new_data).reset_index(drop=True)
    data = df.to_numpy()
print("4\n")
dfcol2 = df[df['col2']%100 <= 24]
dfcol2_25 = df[df['col2']%100 >= 25]
# 记录时间
end_time4 = time.time()

# 输出每行花费的时间
print(f"Time elapsed for current line: {end_time4 - end_time3:.2f} seconds")

# 将处理后的数据写入新的text文件中
with open(output_path, "w") as f:
    for row in data:
        f.write("\t".join(map(str, row)) + "\n")
end_time5 = time.time()
print(f"Time elapsed for current line: {end_time5 - end_time4:.2f} seconds\n")
print(f"Time elapsed for current line: {end_time5 - start_time:.2f} seconds\n")
print("File processed successfully!")