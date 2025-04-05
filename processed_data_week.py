#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:08:16 2023

@author: zhongboan
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
input_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data01~24/processed_Week01~24_File2.txt"
output_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_week_data_week/processed_week_data_week_File2.txt"

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
print(f"Time elapsed for data reading: {end_time2 - end_time1:.2f} seconds\n")
data = np.array(data)
# 将数据转换为 Pandas DataFrame 格式
df = pd.DataFrame(data, columns=['col1', 'col2', 'col3'])
# 对数据进行处理
# 对数据进行分组和排序
df = df.groupby('col1').apply(lambda x: x.sort_values('col2')).reset_index(drop=True)
df = df.sort_values(by=['col1', 'col2']).reset_index(drop=True)

# 对 col3 每两个值相加
col3_values = df['col3'].values
result = []

for i in range(0, len(col3_values), 7*24):
    summed_value = 0  # 重置 summed_value
    for j in range(0, 7*24):
        summed_value += col3_values[i + j]
        result.append(summed_value)

# 将 col3 的计算结果赋值给 DataFrame
df['col3'] = result
df = df[(df['col2'] % 100)% 24 == 0]
df = df[((df['col2'] // 100) - 195) % 7 == 6]

data = df.to_numpy()
end_time3 = time.time()
print(f"Time elapsed for data processing: {end_time3 - end_time2:.2f} seconds\n")

# 将数据写入输出文件
with open(output_path, "w") as f:
    for row in data:
        f.write("\t".join(map(str, row)) + "\n")

end_time4 = time.time()
print(f"Time elapsed for data writing: {end_time4 - end_time3:.2f} seconds\n")
print(f"Time elapsed for current line: {end_time4 - start_time:.2f} seconds\n")
print("File processed successfully!")

