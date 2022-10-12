import numpy as np

a=np.arange(0, 240*320*3).reshape((240, 320, 3)) #type:np.ndarray
print(a.shape)
print(a.transpose((0, 1)))