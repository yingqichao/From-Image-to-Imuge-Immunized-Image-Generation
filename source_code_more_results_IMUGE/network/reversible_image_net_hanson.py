# %matplotlib inline
import torch
import torch.nn as nn
import numpy as np
# from encoder.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from decoder.revert_new import Revert
from noise_layers.identity import Identity
from discriminator.discriminator import Discriminator
from encoder.prep_unet import PrepNetwork_Unet
from loss.vgg_loss import VGGLoss
from network.pure_upsample import PureUpsampling
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.crop import Crop
from noise_layers.jpeg_compression import JpegCompression
from encoder.prep_pureUnetNew import Prep_pureUnet
from noise_layers.DiffJPEG import DiffJPEG
from discriminator.GANloss import GANLoss
from discriminator.NLayerDiscriminator import NLayerDiscriminator
from localizer.local_new import Localize
from noise_layers.gaussian import Gaussian
from noise_layers.resize import Resize


class ReversibleImageNetwork_hanson:
    def __init__(self, username, config=GlobalConfig()):
        super(ReversibleImageNetwork_hanson, self).__init__()
        """ Settings """
        self.alpha = 1.0
        self.roundCount = 0.0
        self.res_count = 1.0
        self.config = config
        self.device = self.config.device
        self.username = username
        self.Another = None
        """ Generator Network"""
        #self.pretrain_net = Pretrain_deepsteg(config=config).cuda()
        # self.encoder_decoder = Net(config=config).cuda()
        self.preprocessing_network = Prep_pureUnet(config=config).cuda()
        if torch.cuda.device_count() > 1:
            self.preprocessing_network = torch.nn.DataParallel(self.preprocessing_network)
        # self.hiding_network = HidingNetwork().cuda()
        # self.reveal_network = RevealNetwork().cuda()
        """ Recovery Network """
        # self.revert_network = RevertNew(input_channel=3, config=config).cuda()
        self.revert_network = Revert(config=config).cuda()
        if torch.cuda.device_count() > 1:
            self.revert_network = torch.nn.DataParallel(self.revert_network)
        """Localize Network"""
        # if self.username=="qichao":
        #     self.localizer = LocalizeNetwork(config).cuda()
        # else:
        #     self.localizer = LocalizeNetwork_noPool(config).cuda()
        """Discriminator"""
        # self.discriminator_CoverHidden = Discriminator(config).cuda()
        # if torch.cuda.device_count() > 1:
        #     self.discriminator_CoverHidden = torch.nn.DataParallel(self.discriminator_CoverHidden)
        # self.discriminator_HiddenRecovery = Discriminator(config).cuda()
        self.discriminator_patchHidden = NLayerDiscriminator().cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_patchHidden = torch.nn.DataParallel(self.discriminator_patchHidden)
        self.discriminator_patchRecovery = NLayerDiscriminator().cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_patchRecovery = torch.nn.DataParallel(self.discriminator_patchRecovery)
        # self.cover_label = 1
        # self.encoded_label = 0
        """Vgg"""

        self.vgg_loss = VGGLoss(3, 1, False).cuda()
        if torch.cuda.device_count() > 1:
            self.vgg_loss = torch.nn.DataParallel(self.vgg_loss)

        """Loss"""
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()
        self.criterionGAN = GANLoss().cuda()
        """Localizer"""
        self.localizer = Localize().cuda()
        if torch.cuda.device_count() > 1:
            self.localizer = torch.nn.DataParallel(self.localizer)

        """Optimizer"""
        self.optimizer_localizer = torch.optim.Adam(self.localizer.parameters())
        self.optimizer_preprocessing_network = torch.optim.Adam(self.preprocessing_network.parameters())
        self.optimizer_revert_network = torch.optim.Adam(self.revert_network.parameters())
        # self.optimizer_discrim_CoverHidden = torch.optim.Adam(self.discriminator_CoverHidden.parameters())
        # self.optimizer_discrim_HiddenRecovery = torch.optim.Adam(self.discriminator_HiddenRecovery.parameters())
        self.optimizer_discrim_patchHiddem = torch.optim.Adam(self.discriminator_patchHidden.parameters())
        self.optimizer_discrim_patchRecovery = torch.optim.Adam(self.discriminator_patchRecovery.parameters())

        """Attack Layers"""

        self.cropout_layer = Cropout(config).cuda()
        if torch.cuda.device_count() > 1:
            self.cropout_layer = torch.nn.DataParallel(self.cropout_layer)
        self.noise_layers = []
        self.jpeg_layer = DiffJPEG(256, 256, differentiable=True).cuda()
        self.jpeg_layer_80 = DiffJPEG(256, 256, quality=80, differentiable=True).cuda()
        self.jpeg_layer_70 = DiffJPEG(256, 256, quality=70, differentiable=True).cuda()
        self.jpeg_layer_60 = DiffJPEG(256, 256, quality=60, differentiable=True).cuda()
        self.jpeg_layer_50 = DiffJPEG(256, 256, quality=50, differentiable=True).cuda()
        # self.gaussian = Gaussian().cuda()
        # self.resize_layer = Resize().cuda()
        self.noise_layers.append(self.jpeg_layer_80)
        self.noise_layers.append(self.jpeg_layer_70)
        self.noise_layers.append(self.jpeg_layer_60)
        self.noise_layers.append(self.jpeg_layer_50)
        # self.noise_layers.append(self.gaussian)
        # self.noise_layers.append(self.resize_layer)
        # if torch.cuda.device_count() > 1:
        #     self.jpeg_layer = torch.nn.DataParallel(self.jpeg_layer)
        self.crop_layer = Crop((0.2, 0.5), (0.2, 0.5)).cuda()

        self.jpeg_layer = DiffJPEG(256, 256, quality=80, differentiable=True).cuda()
        if torch.cuda.device_count() > 1:
            self.jpeg_layer = torch.nn.DataParallel(self.jpeg_layer)

        # self.resize_layer = Resize(config, (0.5, 0.7)).cuda()
        # self.gaussian = Gaussian(config).cuda()
        # self.dropout_layer = Dropout(config,(0.4,0.6)).cuda()
        # """DownSampler"""
        # self.downsample256_32 = PureUpsampling(scale=32 / 256).cuda()
        # self.downsample256_64 = PureUpsampling(scale=64 / 256).cuda()
        # self.downsample256_128 = PureUpsampling(scale=128 / 256).cuda()
        # """Upsample"""
        # self.upsample32_256 = PureUpsampling(scale=256 / 32).cuda()
        # self.upsample64_256 = PureUpsampling(scale=256 / 64).cuda()
        # self.upsample128_256 = PureUpsampling(scale=256 / 128).cuda()
        #
        # self.UpsampleBy2 = PureUpsampling(scale=2)
        # self.DownsampleBy2 = PureUpsampling(scale=1/2)

        self.sigmoid = nn.Sigmoid()

    def getVggLoss(self, marked, cover):
        vgg_on_cov = self.vgg_loss(cover)
        vgg_on_enc = self.vgg_loss(marked)
        loss = self.mse_loss(vgg_on_cov, vgg_on_enc)
        return loss

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def train_on_batch(self, Cover):
        """
            训练方法：先额外训练单个Secret图像向Cover图像嵌入和提取的网络（可以通过读取预训练结果），
            然后将Secret图像送入PrepNetwork(基于Unet)做处理（置乱），送进嵌入和提取网络，
            提取得到的图像再送进RevertNetwork得到近似原图（B），再填充到原图中
            Loss：B与原图的loss，Hidden与原图的loss
        """
        batch_size = Cover.shape[0]
        if self.Another is None:
            print("Got Attack Image")
            self.Another = Cover.clone()
        self.localizer.train()
        self.preprocessing_network.train()
        self.revert_network.train()
        self.discriminator_patchRecovery.train()
        self.discriminator_patchHidden.train()
        self.alpha -= 1/(4*19714)
        self.roundCount += 1/(4*19714)
        self.res_count -= 1 / (4*19714)
        if self.alpha < 0:
            self.alpha = 0
        if self.roundCount > 1:
            self.roundCount = 1
        if self.res_count < 1:
            self.res_count = 1

        with torch.enable_grad():
            """ Run, Train the discriminator"""
            self.optimizer_localizer.zero_grad()
            self.optimizer_preprocessing_network.zero_grad()
            self.optimizer_revert_network.zero_grad()
            self.optimizer_discrim_patchHiddem.zero_grad()
            self.optimizer_discrim_patchRecovery.zero_grad()
            Marked = self.preprocessing_network(Cover)
            Residual = Cover-Marked
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            # Attacked = random_noise_layer(Marked)

            portion_attack, portion_maxPatch = self.config.attack_portion * (1.0 + 0.0 * self.roundCount), \
                                               self.config.crop_size * (1.0 + 0.0 * self.roundCount)
            Cropped_out, CropoutWithCover, cropout_mask = self.cropout_layer(Marked, Cover=self.Another,
                                                                             require_attack=portion_attack,
                                                                             max_size=portion_maxPatch)
            Cropped_out_jpeg = random_noise_layer(Cropped_out)
            CropoutWithCover_jpeg = random_noise_layer(CropoutWithCover)
            loss_cover = self.getVggLoss(Marked, Cover)
            loss_coverPSNR = self.mse_loss(Marked, Cover) * 100

            """Train Localizer """
            pred_label, loss_localization = None, None
            if loss_cover < (2 + 3 * self.alpha):
                print("Localization Trained")
                pred_label = self.localizer(CropoutWithCover_jpeg.detach())
                loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1),
                                                              cropout_mask[:, 0, :, :]) / portion_attack * 10
                loss_localization.backward()
                self.optimizer_localizer.step()

            if loss_localization is not None and loss_localization<10:
                print("Using real prediction..")
                Recovered = self.revert_network(Cropped_out_jpeg,pred_label.clone().detach()[:, 0, :, :].unsqueeze(1), stage=256) #up_256
            else:
                Recovered = self.revert_network(Cropped_out_jpeg, cropout_mask[:, 0, :, :].unsqueeze(1), stage=256)
            # Up_256 = self.upsample128_256(up_256)
            # Up_recover = up_256 * cropout_mask + Cropped_out * (1 - cropout_mask)
            # Out_64 = up_64 * self.alpha + out_64 * (1 - self.alpha)

            """ Discriminate """

            """Discriminator A"""
            """Patch GAN"""
            ## Patch GAN
            loss_D_A = self.backward_D_basic(self.discriminator_patchHidden, Cover, Marked)
            loss_D_A.backward()
            self.optimizer_discrim_patchHiddem.step()
            """DCGAN"""

            """Discriminator B"""
            """Patch GAN"""
            ## Patch GAN
            loss_D_B = self.backward_D_basic(self.discriminator_patchRecovery, Cover, Recovered)
            loss_D_B.backward()
            self.optimizer_discrim_patchRecovery.step()
            """DCGAN"""

            """Losses"""
            ## Local and Global Loss
            loss_R256_globalPSNR = self.mse_loss(Recovered, Cover) / portion_attack * 100
            loss_R256_global = self.getVggLoss(Recovered, Cover)
            loss_R256_local = self.mse_loss(Recovered*cropout_mask,Cover*cropout_mask)/portion_attack * 100 # Temp

            loss_R256 = (loss_R256_global + loss_R256_local+loss_R256_globalPSNR)

            """Adversary Loss"""
            """DCGAN"""


            """Patch GAN"""
            g_loss_adv_enc = self.criterionGAN(self.discriminator_patchHidden(Marked), True)
            g_loss_adv_recovery = self.criterionGAN(self.discriminator_patchRecovery(Recovered), True)
            """DCGAN"""

            print("Loss on 256: Global {0:.6f} PSNR {3:.6f} Local {1:.6f} Sum {2:.6f}".format(loss_R256_global, loss_R256_local, loss_R256, loss_R256_globalPSNR))
            loss_enc_dec = self.config.hyper_recovery * loss_R256 + \
                            g_loss_adv_recovery * self.config.hyper_discriminator

            if loss_cover>(2+3*self.alpha):
                print("Cover Loss added")
                loss_enc_dec += loss_cover * self.config.hyper_cover + \
                                g_loss_adv_enc * self.config.hyper_discriminator + \
                                loss_coverPSNR * self.config.hyper_cover

            if pred_label is not None:
                pred_label = self.localizer(CropoutWithCover_jpeg)
                loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1),
                                                              cropout_mask[:, 0, :, :]) / portion_attack * 10
                loss_enc_dec += loss_localization * self.config.hyper_localizer
            loss_enc_dec.backward()
            self.optimizer_preprocessing_network.step()
            self.optimizer_revert_network.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': loss_localization.item() if pred_label is not None else 1.0,
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_R256.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }

        return losses, (Marked, Recovered, CropoutWithCover_jpeg, pred_label, Residual)

    def test_local(self, Cover):
        batch_size = Cover.shape[0]
        self.localizer.train()
        self.preprocessing_network.train()
        if self.Another is None:
            print("Got Attack Image")
            self.Another = Cover.clone()
        with torch.enable_grad():
            Residual = self.preprocessing_network(Cover)
            Marked = self.res_count * Residual + Cover
            Cropped_out, CropoutWithCover, cropout_mask = self.cropout_layer(Marked, Cover=self.Another,
                                                                             require_attack=0.2,
                                                                             max_size=0.2)

            pred_label = self.localizer(CropoutWithCover)
            print(pred_label.shape)
            loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1), cropout_mask[:, 0, :, :])
            print(loss_localization.item())

        pred_label = self.sigmoid(pred_label)
        return pred_label, CropoutWithCover

    def save_state_dict_all(self, path):
        torch.save(self.revert_network.state_dict(), path + '_revert_network.pkl')
        print("Successfully Saved: " + path + '_revert_network.pkl')
        torch.save(self.preprocessing_network.state_dict(), path + '_prep_network.pkl')
        print("Successfully Saved: " + path + '_prep_network.pkl')
        torch.save(self.localizer.state_dict(), path + '_localizer.pkl')
        print("Successfully Saved: " + path + '_localizer.pkl')
        torch.save(self.discriminator_patchRecovery.state_dict(), path + '_discriminator_patchRecovery.pkl')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pkl')
        torch.save(self.discriminator_patchHidden.state_dict(), path + '_discriminator_patchHidden.pkl')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pkl')

    def save_model(self, path):
        """Saving"""
        # torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
        """"""
        torch.save({'state_dict': self.revert_network.state_dict()}, path + '_revert_network.pth.tar')
        print("Successfully Saved: " + path + '_revert_network.pth.tar')
        torch.save({'state_dict': self.preprocessing_network.state_dict()}, path + '_prep_network.pth.tar')
        print("Successfully Saved: " + path + '_prep_network.pth.tar')
        torch.save({'state_dict': self.discriminator_patchRecovery.state_dict()},
                   path + '_discriminator_patchRecovery.pth.tar')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pth.tar')
        torch.save({'state_dict': self.discriminator_patchHidden.state_dict()},
                   path + '_discriminator_patchHidden.pth.tar')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pth.tar')
        torch.save({'state_dict': self.localizer.state_dict()}, path + '_localizer.pth.tar')
        print("Successfully Saved: " + path + '_localizer.pth.tar')

    def load_state_dict_all(self, path):

        self.preprocessing_network.load_state_dict(torch.load(path + '_prep_network.pkl'))
        print(self.preprocessing_network)
        if torch.cuda.device_count() > 1:
            self.preprocessing_network = torch.nn.DataParallel(self.preprocessing_network)
        print("Successfully Loaded: " + path + '_prep_network.pkl')
        self.revert_network.load_state_dict(torch.load(path + '_revert_network.pkl'))
        print(self.revert_network)
        if torch.cuda.device_count() > 1:
            self.revert_network = torch.nn.DataParallel(self.revert_network)
        print("Successfully Loaded: " + path + '_revert_network.pkl')
        self.localizer.load_state_dict(torch.load(path + '_localizer.pkl'))
        print("Successfully Loaded: " + path + '_localizer.pkl')
        if torch.cuda.device_count() > 1:
            self.localizer = torch.nn.DataParallel(self.localizer)
        self.discriminator_patchRecovery.load_state_dict(torch.load(path + '_discriminator_patchRecovery.pkl'))
        if torch.cuda.device_count() > 1:
            self.discriminator_patchRecovery = torch.nn.DataParallel(self.discriminator_patchRecovery)
        print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pkl')
        self.discriminator_patchHidden.load_state_dict(torch.load(path + '_discriminator_patchHidden.pkl'))
        if torch.cuda.device_count() > 1:
            self.discriminator_patchHidden = torch.nn.DataParallel(self.discriminator_patchHidden)
        print("Successfully Loaded: " + path + '_discriminator_patchHidden.pkl')

    def load_model(self, path):
        """Loading"""
        # model = describe_model()
        # checkpoint = torch.load('checkpoint.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        """"""
        print("Reading From: " + path + '_localizer.pth.tar')
        checkpoint = torch.load(path + '_localizer.pth.tar')
        self.localizer.load_state_dict(checkpoint['state_dict'])
        print(self.localizer)
        print("Successfully Loaded: " + path + '_localizer.pth.tar')
        if torch.cuda.device_count() > 1:
            self.localizer = torch.nn.DataParallel(self.localizer)

        checkpoint = torch.load(path + '_prep_network.pth.tar')
        self.preprocessing_network.load_state_dict(checkpoint['state_dict'])
        print(self.preprocessing_network)
        print("Successfully Loaded: " + path + '_prep_network.pth.tar')
        if torch.cuda.device_count() > 1:
            self.preprocessing_network = torch.nn.DataParallel(self.preprocessing_network)

        checkpoint = torch.load(path + '_revert_network.pth.tar')
        self.revert_network.load_state_dict(checkpoint['state_dict'])
        print(self.revert_network)
        print("Successfully Loaded: " + path + '_revert_network.pth.tar')
        if torch.cuda.device_count() > 1:
            self.revert_network = torch.nn.DataParallel(self.revert_network)

        checkpoint = torch.load(path + '_discriminator_patchRecovery.pth.tar')
        self.discriminator_patchRecovery.load_state_dict(checkpoint['state_dict'])
        print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pth.tar')
        if torch.cuda.device_count() > 1:
            self.discriminator_patchRecovery = torch.nn.DataParallel(self.discriminator_patchRecovery)

        checkpoint = torch.load(path + '_discriminator_patchHidden.pth.tar')
        self.discriminator_patchHidden.load_state_dict(checkpoint['state_dict'])
        print("Successfully Loaded: " + path + '_discriminator_patchHidden.pth.tar')
        if torch.cuda.device_count() > 1:
            self.discriminator_patchHidden = torch.nn.DataParallel(self.discriminator_patchHidden)

    def load_model_old(self, path):
        self.localizer = torch.load(path + '_localizer.pth')
        print("Successfully Loaded: " + path + '_localizer.pth')
        if torch.cuda.device_count() > 1:
            self.localizer = torch.nn.DataParallel(self.localizer)

        self.preprocessing_network = torch.load(path + '_prep_network.pth')
        print(self.preprocessing_network)
        print("Successfully Loaded: " + path + '_prep_network.pth')
        if torch.cuda.device_count() > 1:
            self.preprocessing_network = torch.nn.DataParallel(self.preprocessing_network)

        self.revert_network = torch.load(path + '_revert_network.pth')
        print(self.revert_network)
        print("Successfully Loaded: " + path + '_revert_network.pth')
        if torch.cuda.device_count() > 1:
            self.revert_network = torch.nn.DataParallel(self.revert_network)

        self.discriminator_patchRecovery = torch.load(path + '_discriminator_patchRecovery.pth')
        print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pth')
        if torch.cuda.device_count() > 1:
            self.discriminator_patchRecovery = torch.nn.DataParallel(self.discriminator_patchRecovery)

        self.discriminator_patchHidden = torch.load(path + '_discriminator_patchHidden.pth')
        print("Successfully Loaded: " + path + '_discriminator_patchHidden.pth')
        if torch.cuda.device_count() > 1:
            self.discriminator_patchHidden = torch.nn.DataParallel(self.discriminator_patchHidden)

    def save_model_old(self, path):
        torch.save(self.revert_network, path + '_revert_network.pth')
        print("Successfully Saved: " + path + '_revert_network.pth')
        torch.save(self.preprocessing_network, path + '_prep_network.pth')
        print("Successfully Saved: " + path + '_prep_network.pth')
        torch.save(self.discriminator_patchRecovery, path + '_discriminator_patchRecovery.pth')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pth')
        torch.save(self.discriminator_patchHidden, path + '_discriminator_patchHidden.pth')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pth')
        torch.save(self.localizer, path + '_localizer.pth')
        print("Successfully Saved: " + path + '__localizer.pth')
