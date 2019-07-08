# import numpy as np

# x = np.arange(n)
# # x = np.tile(x,(n,1))
# # x_pad = np.pad(x, ((1,1), (1,1)), 'constant', constant_values=(0, 0))
# print(x)
# # print(x_pad)
#
# from numpy.lib.stride_tricks import as_strided
#
# x_strided = as_strided(x, (n, n), (0, 8))
#
# print(x_strided)

import torch
import torch.nn.functional as F
import torch.nn as nn
import math
n = 5
k = 3

print("1D")
x = torch.arange(n)
print(x)
x_strided = x.as_strided((n - k + 1, k), (1, 1))
print(x_strided)


print("2D")
x = torch.arange(n**2).view(n,n)
print(x)
x_strided = x.as_strided((n - k + 1, n - k + 1, k, k), (n, 1, n, 1))
print(x_strided)
print(x)


print("4D")
b, c, h, w = 2, 2, 4, 5
k = 3
p = 1
x = torch.arange(b*c*h*w).view(b, c, h, w)
print(x)
x_strided = x.as_strided((b, c, h - k + 1, w - k + 1, k, k), (c*h*w, h*w, w, 1, w, 1))
# print(x_strided)
print(x_strided.shape)
print(x_strided[-1,-1,-1,-1,:,:])


print("4D : Unfold")
b, c, h, w = 2, 3, 4, 5
k = 3
p = 1
x = torch.arange(b*c*h*w, dtype=torch.float).view(b, c, h, w)
print(x)
pad = math.floor(k/2)
print("padding={pad}")
unfold = nn.Unfold(kernel_size=(k,k), padding=pad)
x_unfold = unfold(x).view(b, c, k, k, h, w)
print(x_unfold.shape)
print(x_unfold[-1,0,:,:,-1,-1])
#size, stride
