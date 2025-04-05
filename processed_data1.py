#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:08:58 2023

@author: zhongboan
"""

import numpy as np
import os
import errno
import sys
from tqdm import tqdm
import time

# 刷新stdout缓冲区，以便及时输出信息
sys.stdout.flush()
# 指定路徑
input_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/File1.txt"
output_path = "/Users/zhongboan/Desktop/python/CER Electricity Revised March 2012/processed_File1.txt"

# 檢查輸出檔案所在的目錄是否存在，如果不存在，則創建它
if not os.path.exists(os.path.dirname(output_path)):
    try:
        os.makedirs(os.path.dirname(output_path))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# 打開並讀取text檔案
with open(input_path, "r") as f:
    lines = f.readlines()

data = []
# 將數據分割成三個部分，並將其存儲在一個列表中
for line in lines:
    line = line.strip().split()
    data.append([int(line[0]), int(line[1]), float(line[2])])
    # 记录时间
    start_time = time.time()
# 记录时间
start_time = time.time()


# 將數據轉換為numpy數組
data = np.array(data)

# 對數據進行縮放或正規化等等
# 提取第二列并计算余数
col2 = np.array(data)[:, 1] % 100

# 找到所有连续的01到48的位置
positions = []
for i in range(len(col2) - 47):
    if all(col2[i+j] == j+1 for j in range(48)):
        positions.append(i)

end_time = time.time()

# 输出每行花费的时间
print(f"Time elapsed for current line: {end_time - start_time:.2f} seconds")

# 將處理後的數據寫入一個新的text檔案中
with open(output_path, "w") as f:
    for row in data:
        f.write(f"{row[0]} {row[1]} {row[2]}\n")