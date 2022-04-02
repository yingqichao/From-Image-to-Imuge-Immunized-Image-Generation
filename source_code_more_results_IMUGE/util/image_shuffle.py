import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import torch
from torchvision import utils


def imshow(input_img, text, std, mean):
    '''Prints out an image given in tensor format.'''
    imgs_tsor = torch.cat(input_img, 0)
    img = utils.make_grid(imgs_tsor)

    npimg = img.detach().cpu().numpy()
    if img.shape[0] == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(text)
    plt.show()
    return

# 随机数置乱
img = cv2.imread("F:\\ReversibleImage_project2\\sample\\train_coco\\train\\000000000009.jpg",1)
Data_img = img.transpose((2,0,1))
Data_img = np.float32(Data_img/255)
# device = torch.device("cuda")
# imgtsr = torch.tensor(Data_img)
ROWS,COLS = Data_img.shape[1], Data_img.shape[2]
idx = [i for i in range(ROWS*COLS)]
np.random.seed(0)
np.random.shuffle(idx)
print(idx[:100])
plt.imshow(np.transpose(Data_img, (1, 2, 0)))
plt.title('Original')
plt.show()
shuffled = np.zeros_like(Data_img)
sum = 0
for i in range(ROWS):
    for j in range(COLS):
        cvt_row,cvt_col = int(idx[sum]/COLS),idx[sum]%COLS
        shuffled[0][cvt_row][cvt_col] = Data_img[0][i][j]
        shuffled[1][cvt_row][cvt_col] = Data_img[1][i][j]
        shuffled[2][cvt_row][cvt_col] = Data_img[2][i][j]
        sum+=1

plt.imshow(np.transpose(shuffled, (1, 2, 0)))
plt.title('Shuffled')
plt.show()
shuffled_back = np.zeros_like(Data_img)
sum = 0
for i in range(ROWS):
    for j in range(COLS):
        cvt_row, cvt_col = int(idx[sum] / COLS), idx[sum] % COLS
        shuffled_back[0][i][j] = shuffled[0][cvt_row][cvt_col]
        shuffled_back[1][i][j] = shuffled[1][cvt_row][cvt_col]
        shuffled_back[2][i][j] = shuffled[2][cvt_row][cvt_col]
        sum += 1


plt.imshow(np.transpose(shuffled_back, (1, 2, 0)))
plt.title('Shuffled Back')
plt.show()
