# toooest %matplotlib inline
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch import utils
from torchvision import datasets

from config import GlobalConfig
from network.reversible_image_net_hansonRerun256 import ReversibleImageNetwork_hanson
from util import util

class main_hansonRerun256:
    def __init__(self):
        self.config = GlobalConfig()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print("torch.distributed.is_available: "+str(torch.distributed.is_available()))
        # dist.init_process_group(backend='nccl', init_method='env://')

        print("Device Count: {0}".format(torch.cuda.device_count()))
        # Hyper Parameters

        torch.set_printoptions(profile="full")

        self.another = None
        self.train_covers = None

        if not os.path.exists(self.config.MODELS_PATH):
            os.mkdir(self.config.MODELS_PATH)


    def train(self, net, train_loader,another_loader=None, config=GlobalConfig()):
        """ 到这里位置，第一阶段训练已经完成，不然这个函数运行不起来 """

        train_loss_localization, train_loss_cover, train_loss_recover, \
            train_loss_discriminator_enc, train_loss_discriminator_recovery = [], [], [], [], []
        hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, \
            hist_loss_discriminator_recovery = [], [], [], [], []
        for epoch in range(self.config.num_epochs):
            # train
            for idx, train_batch in enumerate(train_loader):
                # 其他图像
                if self.train_covers is None:
                    print("loading irrelevant images..")
                    self.another, _ = iter(another_loader).__next__()
                    self.another = self.another.cuda()
                else:
                    print("using previous images as irrelevant images..")
                    self.another  = self.train_covers

                data, _ = train_batch
                self.train_covers = data.cuda()
                losses, outputA, outputB = net.train_on_batch(self.train_covers,self.another)
                x_hidden, x_recover, x_attacked, pred_label, residual = outputA
                x_recoverB, x_attackedB, x_predB = outputB
                # losses
                train_loss_discriminator_enc.append(losses['loss_discriminator_enc'])
                train_loss_discriminator_recovery.append(losses['loss_discriminator_recovery'])
                train_loss_localization.append(losses['loss_localization'])
                train_loss_cover.append(losses['loss_cover'])
                train_loss_recover.append(losses['loss_recover'])

                str = 'Net 1 Epoch {0}/{1} Training: Batch {2}/{3}. Total Loss {4:.4f}, Localization Loss {5:.4f}, ' \
                      'Cover Loss {6:.4f}, Recover Loss {7:.4f}, Adversial Cover Loss {8:.4f}, Adversial Recovery Loss {9:.4f}' \
                    .format(epoch, self.config.num_epochs, idx + 1, len(train_loader), losses['loss_sum'],
                            losses['loss_localization'], losses['loss_cover'], losses['loss_recover'],
                            losses['loss_discriminator_enc'], losses['loss_discriminator_recovery'])

                print(str)
                if idx % 10240 == 10239:
                    net.save_model(self.config.MODELS_PATH + 'Epoch N{0} Batch {1}'.format((epoch + 1),idx))
                    net.save_model_old(self.config.MODELS_PATH + 'Epoch N{0} Batch {1}'.format((epoch + 1),idx))
                    net.save_state_dict_all(self.config.MODELS_PATH + 'Epoch N{0} Batch {1}'.format((epoch + 1),idx))
                if idx % 128 == 127:
                    for i in range(x_recover.shape[0]):
                        # util.save_images(p7_final[i].cpu(),
                        #                  'epoch-{0}-recovery-batch-{1}-{2}_after7.bmp'.format(epoch, idx, i),
                        #                  './Images/recovery',
                        #                  std=config.std,
                        #                  mean=config.mean)
                        util.save_images(residual[i].cpu(),
                                         'epoch-{0}-residual-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                         './Images/Residual',
                                         std=config.std,
                                         mean=config.mean)
                        util.save_images(x_attacked[i].cpu(),
                                         'epoch-{0}-covers-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                         './Images/attacked',
                                         std=config.std,
                                         mean=config.mean)
                        util.save_images(x_attackedB[i].cpu(),
                                         'epoch-{0}-covers-batch-{1}-{2}-B.bmp'.format(epoch, idx, i),
                                         './Images/attacked',
                                         std=config.std,
                                         mean=config.mean)
                        if pred_label is not None:
                            util.save_images(pred_label[i].cpu(),
                                             'epoch-{0}-covers-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                             './Images/localized')
                            util.save_images(x_predB[i].cpu(),
                                             'epoch-{0}-covers-batch-{1}-{2}-B.bmp'.format(epoch, idx, i),
                                             './Images/localized')
                        util.save_images(x_recover[i].cpu(),
                                         'epoch-{0}-recovery-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                         './Images/recovery',
                                         std=config.std,
                                         mean=config.mean)
                        util.save_images(x_recoverB[i].cpu(),
                                         'epoch-{0}-recovery-batch-{1}-{2}-B.bmp'.format(epoch, idx, i),
                                         './Images/recovery',
                                         std=config.std,
                                         mean=config.mean)
                        util.save_images(x_hidden[i].cpu(),
                                         'epoch-{0}-hidden-batch-{1}-{2}.bmp'.format(epoch, idx, i),
                                         './Images/hidden',
                                         std=config.std,
                                         mean=config.mean)
                        util.save_images(self.train_covers[i].cpu(),
                                         'epoch-{0}-covers-batch-{1}-{2}.bmp'.format(epoch, idx, i),
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
            net.save_model(self.config.MODELS_PATH + 'Epoch N{}'.format(epoch + 1))
            net.save_model_old(self.config.MODELS_PATH + 'Epoch N{}'.format(epoch + 1))
            net.save_state_dict_all(self.config.MODELS_PATH + 'Epoch N{}'.format(epoch + 1))
            # Prints epoch average loss
            print('Epoch [{0}/{1}], Average_loss: Localization Loss {2:.4f}, Cover Loss {3:.4f}, Recover Loss {4:.4f}, '
                  'Adversial Cover Loss {5:.4f}, Adversial Recovery Loss {6:.4f}'.format(
                epoch + 1, self.config.num_epochs, mean_train_loss_localization, mean_train_loss_cover, mean_train_loss_recover,
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
    def run(self):
        net = ReversibleImageNetwork_hanson(username="hanson", config=self.config)
        transform = transforms.Compose([
            transforms.Resize(self.config.Width),
            transforms.RandomCrop(self.config.Width),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean,
                                 std=self.config.std)
        ])
        # Creates training set
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TRAIN_PATH,
                transform), batch_size=self.config.train_batch_size, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)

        another_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                self.config.TRAIN_PATH,
                transform), batch_size=self.config.train_batch_size, num_workers=4,
            pin_memory=True, shuffle=True, drop_last=True)

        # net.load_state_dict_all(MODELS_PATH + 'Epoch N17')
        net.load_state_dict_all(self.config.MODELS_PATH + 'Epoch N3')
        # net.load_state_dict_all(MODELS_PATH + 'Epoch N1')
        net, hist_loss_localization, hist_loss_cover, hist_loss_recover, hist_loss_discriminator_enc, hist_loss_discriminator_recovery \
            = self.train(net, train_loader,another_loader, self.config)

if __name__ == '__main__':
    mainClass = main_hansonRerun256()
    mainClass.run()
