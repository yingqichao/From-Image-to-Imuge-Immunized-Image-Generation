# toooest %matplotlib inline
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import utils
from torchvision import datasets
# from network.localize_net import Localize_hanson
from util import util
from config import GlobalConfig
# from network.reversible_image_net_hide import RNet_test
from network.reversible_image_net_hanson import ReversibleImageNetwork_hanson

if __name__ =='__main__':
    # Setting
    config = GlobalConfig()
    isSelfRecovery = True
    skipTraining = config.skipTraining

    device = config.device
    print(device)
    # Hyper Parameters
    num_epochs = 1
    train_batch_size = config.train_batch_size
    test_batch_size = config.test_batch_size
    learning_rate = config.learning_rate
    use_Vgg = config.useVgg
    use_dataset = config.use_dataset
    # beta = config.beta
    # if use_Vgg:
    #     beta = 10

    MODELS_PATH = config.MODELS_PATH
    VALID_PATH = config.VALID_PATH
    TRAIN_PATH = config.TRAIN_PATH
    TEST_PATH = config.TEST_PATH

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)


    def train_localizer(net, specific_loader, config):
        """ 到这里位置，第一阶段训练已经完成，不然这个函数运行不起来 """

        train_loss_localization  = []
        hist_loss_localization = []
        for epoch in range(num_epochs):
            # train
            for idx, train_batch in enumerate(specific_loader):
                data, _ = train_batch
                train_covers = data.to(device)
                losses, crop_Predicted = net.train_on_batch(train_covers)
                # losses
                train_loss_localization.append(losses['loss_localization'])


            mean_train_loss_localization = np.mean(train_loss_localization)
            hist_loss_localization.append(mean_train_loss_localization)

            # net.save_state_dict(MODELS_PATH + 'Epoch N{}'.format(epoch + 1))
            # Prints epoch average loss
            print('Epoch [{0}/{1}], Average_loss: Localization Loss {2:.4f}'.format(
                epoch + 1, num_epochs, mean_train_loss_localization))

            # validate
            # for idx, test_batch in enumerate(test_loader):
            #     data, _ = test_batch
            #     test_covers = data.to(device)
            #     losses, output = net.validate_on_batch(test_covers, test_covers)

        return net, hist_loss_localization

    def test(net, train_loader, config):

        train_loss_localization, train_loss_cover, train_loss_recover, \
            train_loss_discriminator_enc, train_loss_discriminator_recovery = [], [], [], [], []
        hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, \
            hist_loss_discriminator_recovery = [], [], [], [], []
        for epoch in range(num_epochs):
            # train
            for idx, train_batch in enumerate(train_loader):
                data, _ = train_batch
                train_covers = data.to(device)
                losses, output = net.test_on_batch(train_covers)
                x_hidden, x_recover, x_attacked, pred_label = output
                # losses
                train_loss_discriminator_enc.append(losses['loss_discriminator_enc'])
                train_loss_discriminator_recovery.append(losses['loss_discriminator_recovery'])
                train_loss_localization.append(losses['loss_localization'])
                train_loss_cover.append(losses['loss_cover'])
                train_loss_recover.append(losses['loss_recover'])

                str = 'Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. Total Loss {4:.4f}, Localization Loss {5:.4f}, ' \
                      'Cover Loss {6:.4f}, Recover Loss {7:.4f}, Adversial Cover Loss {8:.4f}, Adversial Recovery Loss {9:.4f}' \
                    .format(epoch, num_epochs, idx + 1, len(train_loader), losses['loss_sum'],
                            losses['loss_localization'], losses['loss_cover'], losses['loss_recover'],
                            losses['loss_discriminator_enc'], losses['loss_discriminator_recovery'])

                print(str)

                for i in range(x_recover.shape[0]):
                    # util.save_images(p7_final[i].cpu(),
                    #                  'epoch-{0}-recovery-batch-{1}-{2}_after7.png'.format(epoch, idx, i),
                    #                  './Images/recovery',
                    #                  std=config.std,
                    #                  mean=config.mean)
                    util.save_images(pred_label[i].cpu(),
                                     'epoch-{0}-localize-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/localized', )
                    util.save_images(x_attacked[i].cpu(),
                                     'epoch-{0}-covers-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/attacked',
                                     std=config.std,
                                     mean=config.mean)
                    util.save_images(x_recover[i].cpu(),
                                     'epoch-{0}-recovery-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/recovery',
                                     std=config.std,
                                     mean=config.mean)
                    util.save_images(x_hidden[i].cpu(),
                                     'epoch-{0}-hidden-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/hidden',
                                     std=config.std,
                                     mean=config.mean)
                    util.save_images(train_covers[i].cpu(),
                                     'epoch-{0}-covers-batch-{1}-{2}.png'.format(epoch, idx, i),
                                     './Images/cover',
                                     std=config.std,
                                     mean=config.mean)

            mean_train_loss_discriminator_enc = np.mean(train_loss_discriminator_enc)
            mean_train_loss_discriminator_recovery = np.mean(train_loss_discriminator_recovery)
            mean_train_loss_localization = np.mean(train_loss_localization)
            mean_train_loss_cover = np.mean(train_loss_cover)
            mean_train_loss_recover = np.mean(train_loss_recover)
            hist_loss_discriminator_enc.append(mean_train_loss_discriminator_enc)
            hist_loss_cover.append(mean_train_loss_cover)
            hist_loss_localization.append(mean_train_loss_localization)
            hist_loss_recover.append(mean_train_loss_recover)
            # net.save_model(MODELS_PATH + 'Epoch N{}'.format(epoch + 1))
            # Prints epoch average loss
            print('Epoch [{0}/{1}], Average_loss: Localization Loss {2:.4f}, Cover Loss {3:.4f}, Recover Loss {4:.4f}, '
                  'Adversial Cover Loss {5:.4f}, Adversial Recovery Loss {6:.4f}'.format(
                epoch + 1, num_epochs, mean_train_loss_localization, mean_train_loss_cover, mean_train_loss_recover,
                mean_train_loss_discriminator_enc, mean_train_loss_discriminator_recovery
            ))

            # validate
            # for idx, test_batch in enumerate(test_loader):
            #     data, _ = test_batch
            #     test_covers = data.to(device)
            #     losses, output = net.validate_on_batch(test_covers, test_covers)

        return net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery

    # ------------------------------------ Begin ---------------------------------------
    # Creates net object
    net = ReversibleImageNetwork_hanson(username="hanson", config=config)

    # Creates training set
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            TEST_PATH,
            transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.mean,
                                     std=config.std),

            ])), batch_size=1, num_workers=1,
        pin_memory=True, shuffle=True, drop_last=True)

    # Creates test set
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(
    #         TEST_PATH,
    #         transforms.Compose([
    #             transforms.Scale(config.Width),
    #             transforms.RandomCrop(config.Width),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=config.mean,
    #                                  std=config.std)
    #         ])), batch_size=test_batch_size, num_workers=1,
    #     pin_memory=True, shuffle=True, drop_last=True)

    """ Begin Pre-Training """
    # if not config.skipPreTraining:
    #     net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery \
    #         = pretrain(net, train_loader, config)
    #     # Plot loss through epochs
    #     util.plt_plot(hist_loss_cover)
    #     util.plt_plot(hist_loss_recover)
    #     util.plt_plot(hist_loss_discriminator_enc)
    #     util.plt_plot(hist_loss_discriminator_recovery)
    # else:
    #     net.load_state_dict_pretrain(MODELS_PATH)
        # net.load_state_dict_Discriminator(torch.load(MODELS_PATH + 'Epoch N' + config.loadfromEpochNum))

    if not config.skipMainTraining:
        net.load_model(MODELS_PATH + 'Epoch N8')
        net.load_localizer(MODELS_PATH + 'Epoch N1')
        # net.load_state_dict_all(MODELS_PATH + 'Epoch N1')
        net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery \
            = test(net, test_loader, config)
        # Plot loss through epochs
        # util.plt_plot(hist_loss_cover)
        # util.plt_plot(hist_loss_recover)
        # util.plt_plot(hist_loss_discriminator_enc)
        # util.plt_plot(hist_loss_discriminator_recovery)
