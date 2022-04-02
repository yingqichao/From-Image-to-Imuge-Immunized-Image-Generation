# %matplotlib inline
import numpy as np
import torch
import torch.nn as nn

# from encoder.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from decoder.revertRerun256 import Revert
from discriminator.GANloss import GANLoss
from discriminator.NLayerDiscriminator import NLayerDiscriminator
from discriminator.discriminator import Discriminator
from encoder.prep_pureUnet import Prep_pureUnet
from localizer.localizer import Localize
from loss.vgg_loss import VGGLoss
from network.pure_upsample import PureUpsampling
from noise_layers.DiffJPEG import DiffJPEG
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout

class ReversibleImageNetwork_hanson:
    def __init__(self, username, config=GlobalConfig()):
        super(ReversibleImageNetwork_hanson, self).__init__()
        """ Settings """
        self.alpha = 0.0
        self.roundCount = 1.0
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
        self.discriminator_CoverHidden = Discriminator(config).cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_CoverHidden = torch.nn.DataParallel(self.discriminator_CoverHidden)
        # self.discriminator_HiddenRecovery = Discriminator(config).cuda()
        self.discriminator_patchHidden = NLayerDiscriminator().cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_patchHidden = torch.nn.DataParallel(self.discriminator_patchHidden)
        self.discriminator_patchRecovery = NLayerDiscriminator().cuda()
        if torch.cuda.device_count() > 1:
            self.discriminator_patchRecovery = torch.nn.DataParallel(self.discriminator_patchRecovery)
        self.cover_label = 1
        self.encoded_label = 0
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
        self.optimizer_discrim_CoverHidden = torch.optim.Adam(self.discriminator_CoverHidden.parameters())
        # self.optimizer_weird_layer = torch.optim.Adam(self.preprocessing_network.final256.parameters())
        self.optimizer_discrim_patchHiddem = torch.optim.Adam(self.discriminator_patchHidden.parameters())
        self.optimizer_discrim_patchRecovery = torch.optim.Adam(self.discriminator_patchRecovery.parameters())

        """Attack Layers"""
        self.noise_layers = []

        self.cropout_layer = Cropout(config).cuda()
        if torch.cuda.device_count() > 1:
            self.cropout_layer = torch.nn.DataParallel(self.cropout_layer)
        # self.jpeg_layer_100 = DiffJPEG(256, 256, quality=100, differentiable=True).cuda()
        self.jpeg_layer_80 = DiffJPEG(256, 256, quality=80, differentiable=True).cuda()
        self.jpeg_layer_90 = DiffJPEG(256, 256, quality=90, differentiable=True).cuda()
        self.jpeg_layer_70 = DiffJPEG(256, 256, quality=70, differentiable=True).cuda()
        self.jpeg_layer_60 = DiffJPEG(256, 256, quality=60, differentiable=True).cuda()
        self.jpeg_layer_50 = DiffJPEG(256, 256, quality=50, differentiable=True).cuda()

        # if torch.cuda.device_count() > 1:
        #     self.jpeg_layer = torch.nn.DataParallel(self.jpeg_layer)
        self.crop_layer = Crop((0.2, 0.5), (0.2, 0.5)).cuda()
        self.noise_layers.append(self.jpeg_layer_80)
        self.noise_layers.append(self.jpeg_layer_90)
        self.noise_layers.append(self.jpeg_layer_70)
        self.noise_layers.append(self.jpeg_layer_60)
        self.noise_layers.append(self.jpeg_layer_50)
        # self.noise_layers.append(self.jpeg_layer_100)
        # if torch.cuda.device_count() > 1:
        #     self.crop_layer = torch.nn.DataParallel(self.crop_layer)

        # self.resize_layer = Resize(config, (0.5, 0.7)).cuda()
        # self.gaussian = Gaussian(config).cuda()
        # self.dropout_layer = Dropout(config,(0.4,0.6)).cuda()
        """DownSampler"""
        self.downsample256_64 = PureUpsampling(scale=64 / 256).cuda()
        self.downsample256_128 = PureUpsampling(scale=128 / 256).cuda()
        """Upsample"""
        self.upsample64_256 = PureUpsampling(scale=256 / 64).cuda()
        self.upsample128_256 = PureUpsampling(scale=256 / 128).cuda()

        self.UpsampleBy2 = PureUpsampling(scale=2)
        self.DownsampleBy2 = PureUpsampling(scale=1/2)

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
        self.alpha -= 1/(118287)
        self.roundCount += 1/(118287)
        self.res_count -= 1 / (2*10240)
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
            Residual = self.preprocessing_network(Cover)
            Marked = Residual+Cover
            """Cropout Attack"""
            portion_attack, portion_maxPatch = self.config.attack_portion * (1.0 + 0.0 * self.roundCount), \
                                               self.config.crop_size * (1.0 + 0.0 * self.roundCount)
            Cropped_out, CropoutWithCover, cropout_mask = self.cropout_layer(Marked, Cover=self.Another,
                                                                          require_attack=portion_attack,max_size=portion_maxPatch)
            """Further JPEG Attack"""
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            Attacked = self.jpeg_layer_80(Cropped_out)
            # AttackedForLocalizer = self.jpeg_layer_50(CropoutWithCover)
            up_256, out_256 = self.revert_network(Attacked, cropout_mask[:, 0, :, :].unsqueeze(1), stage=256) #up_256
            # Up_256 = self.upsample128_256(up_256)
            # Up_recover = up_256 * cropout_mask + Cropped_out * (1 - cropout_mask)
            Out_256 = up_256 * self.alpha + out_256 * (1 - self.alpha)
            Recovered = Out_256 #*cropout_mask+Cropped_out*(1-cropout_mask)

            """ Discriminate """
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float32)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device, dtype=torch.float32)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device, dtype=torch.float32)
            """Discriminator A"""
            ## Patch GAN
            loss_D_A = self.backward_D_basic(self.discriminator_patchHidden, Cover, Marked)
            loss_D_A.backward()
            self.optimizer_discrim_patchHiddem.step()
            # d_on_cover = self.discriminator_CoverHidden(Cover)
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            # d_loss_on_cover.backward()
            # d_on_encoded = self.discriminator_CoverHidden(Marked.detach())
            # d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            # d_loss_on_encoded.backward()
            # self.optimizer_discrim_CoverHidden.step()
            """Discriminator B"""
            ## Patch GAN
            loss_D_B = self.backward_D_basic(self.discriminator_patchRecovery, Cover, Recovered)
            loss_D_B.backward()
            self.optimizer_discrim_patchRecovery.step()

            """Train Localizer """
            pred_label = self.localizer(Attacked.detach())
            loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1), cropout_mask[:, 0, :, :]) / portion_attack * 100
            loss_localization.backward()
            self.optimizer_localizer.step()
            ## Globally
            # d_on_cover = self.discriminator_HiddenRecovery(Cover)
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            # d_loss_on_cover.backward()
            # d_on_encoded = self.discriminator_HiddenRecovery(Recovered.detach())
            # d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            # d_loss_on_encoded.backward()
            # self.optimizer_discrim_HiddenRecovery.step()
            ## Locally
            # d_loss_on_cover_B = 0
            # for i in range(8):
            #     d_on_cover = self.discriminator_HiddenRecovery(self.crop_layer(Cover))
            #     d_loss_on_cover_B += self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            # d_loss_on_cover_B.backward()
            # d_loss_on_recovery = 0
            # for i in range(8):
            #     d_on_encoded = self.discriminator_HiddenRecovery(self.crop_layer(Recovered.detach()))
            #     d_loss_on_recovery += self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            # d_loss_on_recovery.backward()
            # print(
            #     "-- Adversary B on Cover:{0:.6f},on Recovery:{1:.6f} --".format(d_loss_on_cover_B, d_loss_on_recovery))
            # self.optimizer_discrim_HiddenRecovery.step()

            """Losses"""
            ## Local and Global Loss
            loss_R256_global = self.getVggLoss(Recovered, Cover)
            loss_R256_globalPSNR = self.mse_loss(Recovered,Cover) * 100  # Temp
            loss_R256_local = self.mse_loss(Recovered*cropout_mask, Cover*cropout_mask)/portion_attack * 100 # Temp
            loss_R128_global = self.getVggLoss(self.DownsampleBy2(up_256), self.downsample256_128(Cover))
            loss_R128_local = self.mse_loss(self.DownsampleBy2(up_256)*self.DownsampleBy2(cropout_mask),
                                            self.DownsampleBy2(Cover)*self.DownsampleBy2(cropout_mask))/portion_attack * 100
            print("Loss on Pre: Global {0:.6f} Local {1:.6f}, Current alpha: {2:.6f}"
                  .format(loss_R128_global,loss_R128_local,self.alpha))

            loss_cover = self.getVggLoss(Marked, Cover)
            """Adversary Loss"""
            # d_on_encoded_for_enc = self.discriminator_CoverHidden(Marked)
            # g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            g_loss_adv_enc = self.criterionGAN(self.discriminator_patchHidden(Marked), True)
            g_loss_adv_recovery = self.criterionGAN(self.discriminator_patchRecovery(Recovered), True)
            ## Global
            # d_on_encoded_for_recovery = self.discriminator_HiddenRecovery(Recovered)
            # g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            ## Local
            # g_loss_adv_recover = 0
            # loss_R256_global = 0
            # report_str, max_patch_vgg_loss = '                                     ', 0
            # for i in range(8):
            #     crop_shape = self.crop_layer.get_random_rectangle_inside(Recovered)
            #     Recovered_portion = self.crop_layer(Recovered, shape=crop_shape)
            #     Cover_portion = self.crop_layer(Cover, shape=crop_shape)
            #     # d_on_encoded_for_recovery = self.discriminator_HiddenRecovery(Recovered_portion)
            #     patch_vggLoss = self.getVggLoss(Recovered_portion, Cover_portion)
            #     max_patch_vgg_loss = max(patch_vggLoss.item(), max_patch_vgg_loss)
            #     loss_R256_global += patch_vggLoss
            #     # g_loss_adv_recovery += self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            #     # report_str += "Patch {0:.6f} ".format(patch_vggLoss)
            # loss_R256_global = loss_R256_global * max_patch_vgg_loss / loss_R256_global.item()
            # g_loss_adv_recovery /= 8
            loss_R256 = (loss_R256_global+loss_R256_local+loss_R256_globalPSNR)
            print("Loss on 256: Global VGG {0:.6f} PSNR:{3:.6f} Local {1:.6f} Sum {2:.6f}"
                  .format(loss_R256_global, loss_R256_local, loss_R256, loss_R256_globalPSNR))
            loss_enc_dec = self.config.hyper_recovery * loss_R256
            """Localize Loss"""
            # pred_label = self.localizer(AttackedForLocalizer)
            # loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1), cropout_mask[:, 0, :, :]) / portion_attack * 100

            if loss_cover>(1.5+self.alpha*3.5):
                print("Cover Loss added")
                loss_enc_dec += loss_cover * self.config.hyper_cover  # + loss_mask * self.config.hyper_mask
                loss_enc_dec += g_loss_adv_enc * self.config.hyper_discriminator
            loss_enc_dec += g_loss_adv_recovery * self.config.hyper_discriminator # g_loss_adv_enc * self.config.hyper_discriminator +

            # loss_enc_dec += loss_localization * self.config.hyper_localizer
            loss_enc_dec.backward()
            # self.optimizer_weird_layer.step()
            self.optimizer_preprocessing_network.step()
            self.optimizer_revert_network.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_R256.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }

        return losses, (Marked, Recovered, CropoutWithCover, self.sigmoid(pred_label), Residual)

    def test_local(self, Cover):
        batch_size = Cover.shape[0]
        self.localizer.eval()
        self.preprocessing_network.eval()
        self.revert_network.eval()
        if self.Another is None:
            print("Got Attack Image")
            self.Another = Cover.clone()
        with torch.enable_grad():
            # Residual = self.preprocessing_network(Cover)
            # Marked = self.res_count * Residual + Cover
            random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
            Attacked = self.jpeg_layer_50(Cover)
            # Cropped_out, CropoutWithCover, cropout_mask = self.cropout_layer(Attacked, Cover=self.Another,
            #                                                                  require_attack=0.2,
            #                                                                  max_size=0.2)

            pred_label = self.localizer(Attacked)
            Label = self.sigmoid(pred_label)
            up_256, recovered = self.revert_network(Attacked*(1-Label), Label[:, 0, :, :].unsqueeze(1), stage=256)
            # loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1), cropout_mask[:, 0, :, :])/0.2*10
            # print(loss_localization.item())


        return Label, recovered

    def test_on_batch(self, Cover):
        batch_size = Cover.shape[0]
        self.preprocessing_network.eval()
        self.revert_network.eval()
        if self.Another is None:
            print("Got Attack Image")
            self.Another = Cover.clone()
        with torch.enable_grad():
            """ Run, Train the discriminator"""

            Residual = self.preprocessing_network(Cover)
            Marked = Residual + Cover

            Attacked = self.jpeg_layer_80(Marked)
            # portion_attack, portion_maxPatch = self.config.attack_portion * (0.75 + 0.25 * self.roundCount), \
            #                                    self.config.crop_size * (0.75 + 0.25 * self.roundCount)
            portion_attack, portion_maxPatch = 0.3, 0.3
            Cropped_out, CropoutWithCover, cropout_mask = self.cropout_layer(Attacked, Cover=self.Another,
                                                                             require_attack=portion_attack,
                                                                             max_size=portion_maxPatch)
            up_256, out_256 = self.revert_network(Cropped_out, cropout_mask[:, 0, :, :].unsqueeze(1),
                                                  stage=256)  # up_256
            # Up_256 = self.upsample128_256(up_256)
            # Up_recover = up_256 * cropout_mask + Cropped_out * (1 - cropout_mask)
            Out_256 = up_256 * self.alpha + out_256 * (1 - self.alpha)
            Recovered = Out_256  # *cropout_mask+Cropped_out*(1-cropout_mask)

            """Losses"""
            ## Local and Global Loss
            # loss_R256_local = self.mse_loss(Recovered * cropout_mask, Cover * cropout_mask) / portion_attack * 100
            # loss_R256_global = self.getVggLoss(Recovered, Cover)
            loss_R256_local = self.mse_loss(Recovered*cropout_mask, Cover*cropout_mask)/portion_attack * 100 # Temp
            # loss_R128_global = self.getVggLoss(self.DownsampleBy2(up_256), self.downsample256_128(Cover))
            # loss_R128_local = self.mse_loss(Up_recover*cropout_mask, Cover*cropout_mask)/portion_attack * 100
            # print("Loss on Pre: Global {0:.6f} Local {1:.6f}, Current alpha: {2:.6f}"
            #       .format(loss_R128_global,loss_R128_local,self.alpha))

            loss_cover = self.getVggLoss(Marked, Cover)
            """Adversary Loss"""
            # d_on_encoded_for_enc = self.discriminator_CoverHidden(Marked)
            # g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            # g_loss_adv_enc = self.criterionGAN(self.discriminator_patchHidden(Marked), True)
            g_loss_adv_recovery = self.criterionGAN(self.discriminator_patchRecovery(Recovered), True)
            ## Global
            # d_on_encoded_for_recovery = self.discriminator_HiddenRecovery(Recovered)
            # g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            ## Local
            # g_loss_adv_recover = 0
            loss_R256_global = 0
            report_str, max_patch_vgg_loss = '                                     ', 0
            for i in range(8):
                crop_shape = self.crop_layer.get_random_rectangle_inside(Recovered)
                Recovered_portion = self.crop_layer(noised_image=Recovered, shape=crop_shape)
                Cover_portion = self.crop_layer(Cover, shape=crop_shape)
                # d_on_encoded_for_recovery = self.discriminator_HiddenRecovery(Recovered_portion)
                patch_vggLoss = self.getVggLoss(Recovered_portion, Cover_portion)
                max_patch_vgg_loss = max(patch_vggLoss.item(), max_patch_vgg_loss)
                loss_R256_global += patch_vggLoss
                # g_loss_adv_recovery += self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
                # report_str += "Patch {0:.6f} ".format(patch_vggLoss)
            loss_R256_global = loss_R256_global * max_patch_vgg_loss / loss_R256_global.item()
            # g_loss_adv_recovery /= 8
            loss_R256 = (loss_R256_global+loss_R256_local)/2 # (loss_R256_local + loss_R256_global) / 2
            print("Loss on 256: Global {0:.6f} Local {1:.6f} Sum {2:.6f} Curr res Ratio {3:.6f}".format(loss_R256_global, loss_R256_local, loss_R256, self.res_count))
            loss_enc_dec = self.config.hyper_recovery * loss_R256
            """Localize Loss"""
            pred_label = self.localizer(CropoutWithCover)
            loss_localization = self.bce_with_logits_loss(pred_label.squeeze(1), cropout_mask[:, 0, :, :])

            if loss_cover>2:
                print("Cover Loss added")
                loss_enc_dec += loss_cover * self.config.hyper_cover  # + loss_mask * self.config.hyper_mask
            loss_enc_dec += g_loss_adv_recovery * self.config.hyper_discriminator # g_loss_adv_enc * self.config.hyper_discriminator +
            loss_enc_dec += loss_localization * self.config.hyper_localizer

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_R256.item(),
            'loss_discriminator_enc': 0,
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }

        return losses, (Marked, Recovered, Cropped_out, pred_label)

    def save_state_dict_all(self, path):
        torch.save(self.revert_network.state_dict(), path + '_revert_network.pkl')
        print("Successfully Saved: " + path + '_revert_network.pkl')
        torch.save(self.preprocessing_network.state_dict(), path + '_prep_network.pkl')
        print("Successfully Saved: " + path + '_prep_network.pkl')
        torch.save(self.localizer, path + '_localizer.pkl')
        print("Successfully Saved: " + path + '_localizer.pkl')
        torch.save(self.discriminator_patchRecovery, path + '_discriminator_patchRecovery.pkl')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pkl')
        torch.save(self.discriminator_CoverHidden, path + '_discriminator_CoverHidden.pkl')
        print("Successfully Saved: " + path + '_discriminator_CoverHidden.pkl')


    def save_model(self, path):
        """Saving"""
        # torch.save({'state_dict': model.state_dict()}, 'checkpoint.pth.tar')
        """"""
        torch.save({'state_dict': self.revert_network.state_dict()}, path + '_revert_network.pth.tar')
        print("Successfully Saved: " + path + '_revert_network.pth.tar')
        torch.save({'state_dict': self.preprocessing_network.state_dict()}, path + '_prep_network.pth.tar')
        print("Successfully Saved: " + path + '_prep_network.pth.tar')
        torch.save({'state_dict': self.discriminator_patchRecovery.state_dict()}, path + '_discriminator_patchRecovery.pth.tar')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pth.tar')
        torch.save({'state_dict': self.discriminator_patchHidden.state_dict()},path + '_discriminator_patchHidden.pth.tar')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pth.tar')
        torch.save({'state_dict': self.localizer.state_dict()},path + '_localizer.pth.tar')
        print("Successfully Saved: " + path + '_localizer.pth.tar')

    def load_state_dict_all(self, path):
        # self.discriminator_CoverHidden.load_state_dict(torch.load(path + '_discriminator_CoverHidden.pkl'), strict=False)
        # print("Successfully Loaded: " + path + '_discriminator_CoverHidden.pkl')
        self.preprocessing_network.load_state_dict(torch.load(path + '_prep_network.pkl'))
        print(self.preprocessing_network)
        print("Successfully Loaded: " + path + '_prep_network.pkl')
        self.revert_network.load_state_dict(torch.load(path + '_revert_network.pkl'))
        print(self.revert_network)
        print("Successfully Loaded: " + path + '_revert_network.pkl')
        # self.localizer.load_state_dict(torch.load(path + '_localizer.pkl'), strict=False)
        # print("Successfully Loaded: " + path + '_localizer.pkl')
        # self.discriminator_patchRecovery.load_state_dict(torch.load(path + '_discriminator_patchRecovery.pkl'), strict=False)
        # print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pkl')

    def load_model(self, path):
        """Loading"""
        # model = describe_model()
        # checkpoint = torch.load('checkpoint.pth.tar')
        # model.load_state_dict(checkpoint['state_dict'])
        """"""
        # print("Reading From: " + path + '_localizer.pth.tar')
        # checkpoint = torch.load(path + '_localizer.pth.tar')
        # self.localizer.load_state_dict(checkpoint['state_dict'])
        # print(self.localizer)
        print("Successfully Loaded: " + path + '_localizer.pth.tar')

        checkpoint = torch.load(path + '_prep_network.pth.tar')
        self.preprocessing_network.load_state_dict(checkpoint['state_dict'],strict=False)
        print(self.preprocessing_network)
        print("Successfully Loaded: " + path + '_prep_network.pth.tar')

        checkpoint = torch.load(path + '_revert_network.pth.tar')
        self.revert_network.load_state_dict(checkpoint['state_dict'],strict=False)
        print(self.revert_network)
        print("Successfully Loaded: " + path + '_revert_network.pth.tar')

        checkpoint = torch.load(path + '_discriminator_patchRecovery.pth.tar')
        self.discriminator_patchRecovery.load_state_dict(checkpoint['state_dict'],strict=False)
        print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pth.tar')

        checkpoint = torch.load(path + '_discriminator_CoverHidden.pth.tar')
        self.discriminator_CoverHidden.load_state_dict(checkpoint['state_dict'],strict=False)
        print("Successfully Loaded: " + path + '_discriminator_CoverHidden.pth.tar')

    def save_model_old(self, path):
        torch.save(self.revert_network, path + '_revert_network.pth')
        print("Successfully Saved: " + path + '_revert_network.pth')
        torch.save(self.preprocessing_network, path + '_prep_network.pth')
        print("Successfully Saved: " + path + '_prep_network.pth')
        # torch.save(self.discriminator_patchHidden, path + '_discriminator_patchHidden.pth')
        # print("Successfully Saved: " + path + '_discriminator_patchHidden.pth')
        torch.save(self.discriminator_patchRecovery, path + '_discriminator_patchRecovery.pth')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pth')
        torch.save(self.discriminator_patchHidden, path + '_discriminator_patchHidden.pth')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pth')
        torch.save(self.localizer, path + '_localizer.pth')
        print("Successfully Saved: " + path + '__localizer.pth')

    def load_model_old(self, path):
        # self.localizer = torch.load(path + '_localizer.pth')
        # print("Successfully Loaded: " + path + '_localizer.pth')
        self.preprocessing_network = torch.load(path + '_prep_network.pth')
        print(self.preprocessing_network)
        print("Successfully Loaded: " + path + '_prep_network.pth')
        self.revert_network = torch.load(path + '_revert_network.pth')
        print(self.revert_network)
        print("Successfully Loaded: " + path + '_revert_network.pth')
        self.discriminator_patchRecovery = torch.load(path + '_discriminator_patchRecovery.pth')
        print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pth')
        # self.discriminator_patchHidden = torch.load(path + '_discriminator_patchHidden.pth')
        # print("Successfully Loaded: " + path + '_discriminator_patchHidden.pth')

