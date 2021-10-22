# -*- coding: utf-8 -*-
# 作者 : 王天赐

from tqdm import tqdm
import numpy as np

arr = np.random.random(1000)

for data in tqdm(arr, total=len(arr)):
    data += 1
