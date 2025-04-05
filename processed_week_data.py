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
input_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_data01~48/processed_File6.txt"
output_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data/processed_Week_File6.txt"

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
for line in tqdm(lines):
    line = line.strip().split()
    data.append([float(line[0]), float(line[1]), float(line[2])])
end_time2 = time.time()
data = np.array(data)
# 将数据转换为 Pandas DataFrame 格式
df = pd.DataFrame(data, columns=['col1', 'col2', 'col3'])
# 对 col1 进行分组并按 col2 进行排序
df = df.groupby('col1').apply(lambda x: x.sort_values('col2')).reset_index(drop=True)
print("\n1\n")
col2 = df['col2'] / 100
col2mod = df['col2'] % 100
col2 = col2.astype('int')
first_monday = col2.min()
print("2\n")
end_time2 = time.time()
print(f"Time elapsed for current line: {end_time2 - end_time1:.2f} seconds\n")
col1 = df['col1']
day = 48
week_length = day * 7
positions = []
i = 0
with tqdm(total=len(col2) - week_length + 1, desc="Finding positions") as pbar:
    while i < len(col2) - week_length + 1:
        if col1[i] == col1[i + week_length - 1] and col2mod[i] == 1 and all(col2[i + day * j] == col2[i] + j for j in range(1, 7)) and ((col2[i] - first_monday) % 7 == 0):
            positions.append(i)
            i += week_length
            pbar.update(week_length)
        else:
            i += day
            pbar.update(day)
pos = np.array(positions)
pos = pd.DataFrame(pos)
print("3\n")
end_time3 = time.time()
print(f"Time elapsed for current line: {end_time3 - end_time2:.2f} seconds\n")
if not positions:
    df = pd.DataFrame()
    data = df
else:
    new_data = []
    for start_pos in tqdm(positions, desc="Extracting continuous data"):
        new_data.append(df.iloc[start_pos:start_pos+336])
    df = pd.concat(new_data).reset_index(drop=True)
    data = df.to_numpy()
#dfcol22_25 = df[df['col2']%100 >= 25]
print("4\n")
end_time4 = time.time()
print(f"Time elapsed for current line: {end_time4 - end_time3:.2f} seconds")
with open(output_path, "w") as f:
    for row in data:
        f.write("\t".join(map(str, row)) + "\n")
end_time5 = time.time()
print(f"Time elapsed for current line: {end_time5 - end_time4:.2f} seconds\n")
print(f"Time elapsed for current line: {end_time5 - start_time:.2f} seconds\n")
print("File processed successfully!")