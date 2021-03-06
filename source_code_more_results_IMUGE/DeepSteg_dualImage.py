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

# Directory path
# os.chdir("..")
if __name__ =='__main__':
    cwd = '.'
    device = torch.device("cuda")
    print(device)
    # Hyper Parameters
    num_epochs = 20
    batch_size = 2
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


    def customized_loss(train_output, train_hidden, train_covers,train_secrets, B):
        ''' Calculates loss specified on the paper.'''
        # train_output, train_hidden,train_covers,(secret1,secret2),beta

        loss_cover = torch.nn.functional.mse_loss(train_hidden, train_covers)
        loss_secret = torch.nn.functional.mse_loss(train_output[0], train_secrets[0])
        loss_secret += torch.nn.functional.mse_loss(train_output[1], train_secrets[1])
        loss_secret /= 2
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
        plt.title('Example ' + str(idx) + ', lr=' + str(learning_rate) + ', B=' + str(beta)+' ???????????? ???????????? ???????????? ?????????????????????')
        plt.show()
        return


    def gaussian(tensor, mean=0, stddev=0.1):
        '''Adds random noise to a tensor.'''

        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)

        return tensor + noise


    # Preparation Network (2 conv layers)
    class PrepNetwork(nn.Module):
        def __init__(self):
            super(PrepNetwork, self).__init__()
            self.initialP3 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU())
            self.initialP4 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=4, padding=2),
                nn.ReLU())
            self.initialP5 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.ReLU())
            self.finalP3 = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=3, padding=1),
                nn.ReLU())
            self.finalP4 = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=4, padding=2),
                nn.ReLU())
            self.finalP5 = nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=5, padding=2),
                nn.ReLU())

        def forward(self, p):
            p1 = self.initialP3(p)
            p2 = self.initialP4(p)
            p3 = self.initialP5(p)
            mid = torch.cat((p1, p2, p3), 1)
            p4 = self.finalP3(mid)
            p5 = self.finalP4(mid)
            p6 = self.finalP5(mid)
            out = torch.cat((p4, p5, p6), 1)
            return out


    # Hiding Network (5 conv layers)
    class HidingNetwork(nn.Module):
        def __init__(self):
            super(HidingNetwork, self).__init__()
            self.initialH3 = nn.Sequential(
                nn.Conv2d(96*2+3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU())
            self.initialH4 = nn.Sequential(
                nn.Conv2d(96*2+3, 64, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, padding=2),
                nn.ReLU())
            self.initialH5 = nn.Sequential(
                nn.Conv2d(96*2+3, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.ReLU())
            self.finalH3 = nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=3, padding=1),
                nn.ReLU())
            self.finalH4 = nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=4, padding=2),
                nn.ReLU())
            self.finalH5 = nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=5, padding=2),
                nn.ReLU())
            self.finalH = nn.Sequential(
                nn.Conv2d(192, 3, kernel_size=1, padding=0))

        def forward(self, h):
            h1 = self.initialH3(h)
            h2 = self.initialH4(h)
            h3 = self.initialH5(h)
            mid = torch.cat((h1, h2, h3), 1)
            h4 = self.finalH3(mid)
            h5 = self.finalH4(mid)
            h6 = self.finalH5(mid)
            mid2 = torch.cat((h4, h5, h6), 1)
            out = self.finalH(mid2)
            # out_noise = gaussian(out.data, 0, 0.1)
            return out


    # Reveal Network (2 conv layers)
    class RevealNetwork(nn.Module):
        def __init__(self):
            super(RevealNetwork, self).__init__()
            self.initialR3 = nn.Sequential(
                nn.Conv2d(3, 50, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=3, padding=1),
                nn.ReLU())
            self.initialR4 = nn.Sequential(
                nn.Conv2d(3, 50, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=4, padding=2),
                nn.ReLU())
            self.initialR5 = nn.Sequential(
                nn.Conv2d(3, 50, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=5, padding=2),
                nn.ReLU())
            self.finalR3 = nn.Sequential(
                nn.Conv2d(150, 50, kernel_size=3, padding=1),
                nn.ReLU())
            self.finalR4 = nn.Sequential(
                nn.Conv2d(150, 50, kernel_size=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 50, kernel_size=4, padding=2),
                nn.ReLU())
            self.finalR5 = nn.Sequential(
                nn.Conv2d(150, 50, kernel_size=5, padding=2),
                nn.ReLU())
            self.finalR = nn.Sequential(
                nn.Conv2d(150, 3, kernel_size=1, padding=0))

        def forward(self, r):
            r1 = self.initialR3(r)
            r2 = self.initialR4(r)
            r3 = self.initialR5(r)
            mid = torch.cat((r1, r2, r3), 1)
            r4 = self.finalR3(mid)
            r5 = self.finalR4(mid)
            r6 = self.finalR5(mid)
            mid2 = torch.cat((r4, r5, r6), 1)
            out = self.finalR(mid2)
            return out


    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    # Join three networks in one module
    class Net(nn.Module):
        def __init__(self, config=GlobalConfig()):
            super(Net, self).__init__()
            self.m1 = PrepNetwork().to(device)
            self.m2 = HidingNetwork().to(device)
            self.reveal1 = RevealNetwork().to(device)
            self.reveal2 = RevealNetwork().to(device)
            self.device = config.device
            # Noise Network
            self.jpeg_layer = JpegCompression(self.device)

            # self.cropout_layer = Cropout(config).to(self.device)
            self.gaussian = Gaussian(config).to(self.device)
            self.resize_layer = Resize((0.5, 0.7)).to(self.device)

        def forward(self, cover, secret1, secret2):
            # secret1 = flip(cover, 2).detach()
            # secret2 = flip(cover, 3).detach()
            x_1 = self.m1(secret1)
            x_2 = self.m1(secret2)
            mid = torch.cat((x_1,x_2, cover), 1)
            Hidden = self.m2(mid)
            x_gaussian = self.gaussian(Hidden)
            # x_1_resize = self.resize_layer(x_1_gaussian)
            x_attack = self.jpeg_layer(x_gaussian)
            Recovery1 = self.reveal1(x_attack)
            Recovery2 = self.reveal2(x_attack)
            return Hidden, (Recovery1, Recovery2)


    def train_model(net, train_loader, beta, learning_rate,isSelfRecovery=True):
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


                train_covers = data[:].to(device)
                # train_secrets = data[:]

                # Creates variable from secret and cover images
                secret1 = flip(train_covers, 2).detach()
                secret2 = flip(train_covers, 3).detach()


                # Forward + Backward + Optimize
                optimizer.zero_grad()
                train_hidden, train_output = net(train_covers, secret1, secret2)
                Recover_y, Recover_x = train_output[0], train_output[1]

                # Calculate loss and perform backprop
                train_loss, train_loss_cover, train_loss_secret = customized_loss(train_output, train_hidden,train_covers,(secret1,secret2),beta)
                train_loss.backward()
                optimizer.step()

                # Saves training loss
                train_losses.append(train_loss.data.cpu().numpy())
                loss_history.append(train_loss.data.cpu().numpy())

                if idx % 128 == 127:
                    for i in range(Recover_y.shape[0]):
                        util.save_images(Recover_y[i].cpu(),
                                         'epoch-{0}-recovery-batch-{1}-{2}-y.png'.format(epoch, idx, i),
                                         './Images/pretrain/recovery',
                                         std=config.std,
                                         mean=config.mean)
                        util.save_images(Recover_x[i].cpu(),
                                         'epoch-{0}-recovery-batch-{1}-{2}-x.png'.format(epoch, idx, i),
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

            torch.save(net.state_dict(), MODELS_PATH + '_pretrain_DualImage_Epoch N{}.pkl'.format(epoch + 1))

            mean_train_loss = np.mean(train_losses)

            # Prints epoch average loss
            print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch + 1, num_epochs, mean_train_loss))

        return net, mean_train_loss, loss_history


    # Setting
    isSelfRecovery = True
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

    net, mean_train_loss, loss_history = train_model(net, train_loader, beta, learning_rate, isSelfRecovery)
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
        if not isSelfRecovery:
            test_secret = data[:len(data) // 2]
            test_cover = data[len(data) // 2:]
        else:
            # Self Recovery
            test_secret = data[:]
            test_cover = data[:]


        # Creates variable from secret and cover images
        test_secret = torch.tensor(test_secret, requires_grad=False).to(device)
        test_cover = torch.tensor(test_cover, requires_grad=False).to(device)

        # Compute output
        test_hidden, test_output = net(test_secret, test_cover) # ??????????????????????????????????????????????????????

        # Calculate loss
        test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta)

        #     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))

        #     print (diff_S, diff_C)

        if idx in [1, 2, 3, 4]:
            print('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.data,
                                                                                               loss_secret.data,
                                                                                               loss_cover.data))

            # Creates img tensor
            imgs = [test_secret.data,  test_cover.data, test_hidden.data, test_output.data] # ????????????  ???????????? ???????????? ?????????????????????
            imgs_tsor = torch.cat(imgs, 0)

            # Prints Images
            imshow(utils.make_grid(imgs_tsor), idx + 1, learning_rate=learning_rate, beta=beta)

        test_losses.append(test_loss.data.cpu().numpy())

    mean_test_loss = np.mean(test_losses)

    print('Average loss on test set: {:.2f}'.format(mean_test_loss))