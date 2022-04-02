import cv2
import numpy as np
import numpy.random as random

# 随机数置乱
img = cv2.imread("F:\\ReversibleImage_project2\\sample\\train_coco\\train\\000000000009.jpg",1)


idx = [i for i in range(256*256)]
np.random.seed(0)
np.random.shuffle(idx)
print(idx[:100])

values = [i for i in range(256*256)]
new_values = np.zeros([256*256,1])
idx = [i for i in range(256*256)]
np.random.seed(0)
np.random.shuffle(idx)
for i in range(256*256):
    new_values[idx[i]] = values[i]
revert = np.zeros([256*256,1])
for i in range(256*256):
    revert[i] = new_values[idx[i]]

print()
print(revert[:100])
