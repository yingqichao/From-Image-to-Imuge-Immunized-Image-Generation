# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
from util import util
from torch import utils
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, utils
import torchvision.transforms as transforms
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.resize import Resize
from noise_layers.gaussian import Gaussian
from config import GlobalConfig
from network.reveal import RevealNetwork
from network.prepare import PrepNetwork
from network.hiding import HidingNetwork


def customized_loss(train_output, train_hidden, train_secrets, train_covers, B):
    ''' Calculates loss specified on the paper.'''
    # train_output, train_hidden, train_secrets, train_covers

    loss_cover = torch.nn.functional.mse_loss(train_hidden, train_covers)
    loss_secret = torch.nn.functional.mse_loss(train_output, train_secrets)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret


def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image


def imshow(img, idx, learning_rate, beta):
    '''Prints out an image given in tensor format.'''

    img = denormalize(img, std, mean)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta)+' 隐藏图像 宿主图像 输出图像 提取得到的图像')
    plt.show()
    return


def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''

    noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)

    return tensor + noise





# Join three networks in one module
class Net(nn.Module):
    def __init__(self, config=GlobalConfig()):
        super(Net, self).__init__()
        self.config = config
        self.device = config.device
        self.m1 = PrepNetwork().to(self.device)
        self.m2 = HidingNetwork().to(self.device)
        self.m3 = RevealNetwork().to(self.device)
        # Noise Network
        self.jpeg_layer = JpegCompression(self.device)

        # self.cropout_layer = Cropout(config).to(self.device)
        self.gaussian = Gaussian(config).to(self.device)
        self.resize_layer = Resize((0.5, 0.7)).to(self.device)

    def forward(self, secret, cover):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        Hidden = self.m2(mid)
        x_gaussian = self.gaussian(Hidden)
        # x_1_resize = self.resize_layer(x_1_gaussian)
        x_attack = self.jpeg_layer(x_gaussian)
        Recovery = self.m3(x_attack)
        return Hidden, Recovery


def train_model(net, train_loader, beta, learning_rate):
    # Save optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    config = GlobalConfig()

    loss_history = []
    # Iterate over batches performing forward and backward passes
    for epoch in range(num_epochs):

        # Train mode
        net.train()

        train_losses = []
        # Train one epoch
        for idx, train_batch in enumerate(train_loader):
            data, _ = train_batch

            # Saves secret images and secret covers

            train_covers = data[:len(data) // 2]
            train_secrets = data[len(data) // 2:]


            # Creates variable from secret and cover images
            train_secrets = torch.tensor(train_secrets, requires_grad=False).to(device)
            train_covers = torch.tensor(train_covers, requires_grad=False).to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            train_hidden, train_output = net(train_secrets, train_covers)

            # Calculate loss and perform backprop
            train_loss, train_loss_cover, train_loss_secret = customized_loss(train_output, train_hidden, train_secrets,
                                                                              train_covers, beta)
            train_loss.backward()
            optimizer.step()

            # Saves training loss
            train_losses.append(train_loss.data.cpu().numpy())
            loss_history.append(train_loss.data.cpu().numpy())

            if idx % 128 == 127:
                for i in range(train_output.shape[0]):
                    util.save_images(train_output[i].cpu(),
                                     'epoch-{0}-recovery-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/pretrain/recovery',
                                     std=config.std,
                                     mean=config.mean)
                    util.save_images(train_hidden[i].cpu(),
                                     'epoch-{0}-hidden-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/pretrain/hidden',
                                     std=config.std,
                                     mean=config.mean)
                    util.save_images(train_covers[i].cpu(),
                                     'epoch-{0}-covers-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/pretrain/original',
                                     std=config.std,
                                     mean=config.mean)

            # Prints mini-batch losses
            print('Training: Batch {0}/{1}. Loss of {2:.4f}, cover loss of {3:.4f}, secret loss of {4:.4f}'.format(
                idx + 1, len(train_loader), train_loss.data, train_loss_cover.data, train_loss_secret.data))

        torch.save(net.state_dict(), MODELS_PATH + '_pretrain_Epoch N{}.pkl'.format(epoch + 1))
        torch.save(net.m2.state_dict(), MODELS_PATH + '_pretrain_hiding_Epoch N{}.pkl'.format(epoch + 1))
        torch.save(net.m3.state_dict(), MODELS_PATH + '_pretrain_reveal_Epoch N{}.pkl'.format(epoch + 1))
        mean_train_loss = np.mean(train_losses)

        # Prints epoch average loss
        print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
            epoch + 1, num_epochs, mean_train_loss))

    return net, mean_train_loss, loss_history

if __name__ =='__main__':
    cwd = '.'
    device = torch.device("cuda")
    print(device)
    # Hyper Parameters
    num_epochs = 50
    batch_size = 4
    learning_rate = 0.0001
    beta = 1

    # Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    # TODO: Define train, validation and models
    MODELS_PATH = './output/models/'
    # TRAIN_PATH = cwd+'/train/'
    # VALID_PATH = cwd+'/valid/'
    VALID_PATH = './sample/valid_coco/'
    TRAIN_PATH = './sample/train_coco/'
    TEST_PATH = './sample/test_coco/'
    if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)

    # Creates net object
    net = Net().to(device)
    # Creates training set
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TRAIN_PATH,
            transforms.Compose([
                transforms.Scale(300),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])), batch_size=batch_size, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    # Creates test set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TEST_PATH,
            transforms.Compose([
                transforms.Scale(300),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                     std=std)
            ])), batch_size=1, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    net, mean_train_loss, loss_history = train_model(net, train_loader, beta, learning_rate)
    # Plot loss through epochs
    plt.plot(loss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()
    # net.load_state_dict(torch.load(MODELS_PATH+'Epoch N4.pkl'))

    # Switch to evaluate mode
    net.eval()

    test_losses = []
    # Show images
    for idx, test_batch in enumerate(test_loader):
        # Saves images
        data, _ = test_batch

        # Saves secret images and secret covers

        test_secret = data[:len(data) // 2]
        test_cover = data[len(data) // 2:]

        # Creates variable from secret and cover images
        test_secret = torch.tensor(test_secret, requires_grad=False).to(device)
        test_cover = torch.tensor(test_cover, requires_grad=False).to(device)

        # Compute output
        test_hidden, test_output = net(test_secret, test_cover) # 第一个是输出，第二个是叠加了高斯噪声

        # Calculate loss
        test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta)

        #     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))

        #     print (diff_S, diff_C)

        if idx in [1, 2, 3, 4]:
            print('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.data,
                                                                                               loss_secret.data,
                                                                                               loss_cover.data))

            # Creates img tensor
            imgs = [test_secret.data,  test_cover.data, test_hidden.data, test_output.data] # 隐藏图像  宿主图像 输出图像 提取得到的图像
            imgs_tsor = torch.cat(imgs, 0)

            # Prints Images
            imshow(utils.make_grid(imgs_tsor), idx + 1, learning_rate=learning_rate, beta=beta)

        test_losses.append(test_loss.data.cpu().numpy())

    mean_test_loss = np.mean(test_losses)

    print('Average loss on test set: {:.2f}'.format(mean_test_loss))