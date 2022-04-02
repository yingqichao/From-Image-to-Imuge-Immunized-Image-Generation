# %matplotlib inline
import torch
import torch.nn as nn

# from encoder.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from decoder.revert import Revert
from discriminator.discriminator import Discriminator
from encoder.prep_unet import PrepNetwork_Unet
from loss.vgg_loss import VGGLoss
from network.pure_upsample import PureUpsampling
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.crop import Crop
from discriminator.GANloss import GANLoss
from discriminator.NLayerDiscriminator import NLayerDiscriminator
from encoder.prep_stegano import PrepStegano
from decoder.revert_stegano import RevertStegano
from noise_layers.DiffJPEG import DiffJPEG

class ReversibleImageNetwork_qichao:
    def __init__(self, username, config=GlobalConfig()):
        super(ReversibleImageNetwork_qichao, self).__init__()
        """ Settings """
        self.alpha = 0.9
        self.roundCount = 0.1
        self.config = config
        #self.device = self.config.device
        self.username = username
        """ Generator Network"""
        #self.pretrain_net = Pretrain_deepsteg(config=config).cuda()
        # self.encoder_decoder = Net(config=config).cuda()
        self.preprocessing_network = PrepStegano(config=config).cuda()
        if torch.cuda.device_count()>1:
            self.preprocessing_network = torch.nn.DataParallel(self.preprocessing_network)
        # self.hiding_network = HidingNetwork().cuda()
        # self.reveal_network = RevealNetwork().cuda()
        """ Recovery Network """
        # self.revert_network = RevertNew(input_channel=3, config=config).cuda()
        self.revert_network = RevertStegano(config=config).cuda()
        if torch.cuda.device_count()>1:
            self.revert_network = torch.nn.DataParallel(self.revert_network)
        """Localize Network"""
        # if self.username=="qichao":
        #     self.localizer = LocalizeNetwork(config).cuda()
        # else:
        #     self.localizer = LocalizeNetwork_noPool(config).cuda()
        """Discriminator"""
        # self.discriminator_CoverHidden = Discriminator(config).cuda()
        # if torch.cuda.device_count()>1:
        #     self.discriminator_CoverHidden = torch.nn.DataParallel(self.discriminator_CoverHidden)
        # self.discriminator_HiddenRecovery = Discriminator(config).cuda()
        # if torch.cuda.device_count()>1:
        #     self.discriminator_HiddenRecovery = torch.nn.DataParallel(self.discriminator_HiddenRecovery)
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

        """Loss"""
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()
        self.mse_loss = nn.MSELoss().cuda()
        self.criterionGAN = GANLoss().cuda()
        """Optimizer"""
        self.optimizer_preprocessing_network = torch.optim.Adam(self.preprocessing_network.parameters())
        self.optimizer_revert_network = torch.optim.Adam(self.revert_network.parameters())
        # self.optimizer_discrim_CoverHidden = torch.optim.Adam(self.discriminator_CoverHidden.parameters())
        # self.optimizer_discrim_HiddenRecovery = torch.optim.Adam(self.discriminator_HiddenRecovery.parameters())
        self.optimizer_discrim_patchHiddem = torch.optim.Adam(self.discriminator_patchHidden.parameters())
        self.optimizer_discrim_patchRecovery = torch.optim.Adam(self.discriminator_patchRecovery.parameters())

        """Attack Layers"""
        self.cropout_layer = Cropout(config).cuda()
        self.jpeg_layer = DiffJPEG(256, 256, differentiable=True, quality=80).cuda()
        self.crop_layer = Crop((0.2,0.5),(0.2,0.5)).cuda()
        # self.resize_layer = Resize(config, (0.5, 0.7)).cuda()
        # self.gaussian = Gaussian(config).cuda()
        # self.dropout_layer = Dropout(config,(0.4,0.6)).cuda()
        """DownSampler"""
        # self.downsample256_128 = PureUpsampling(scale=128/ 256).cuda()
        """Upsample"""
        # self.upsample128_256 = PureUpsampling(scale=256 / 128).cuda()

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
        self.preprocessing_network.train()
        self.revert_network.train()
        self.discriminator_patchRecovery.train()
        self.discriminator_patchHidden.train()
        # self.alpha -= 1 / (4 * 10240)
        self.roundCount += 1 / (4 * 10240)
        # if self.alpha < 0:
        #     self.alpha = 0
        if self.roundCount > 1:
            self.roundCount = 1

        with torch.enable_grad():
            """ Run, Train the discriminator"""
            self.optimizer_preprocessing_network.zero_grad()
            self.optimizer_revert_network.zero_grad()
            self.optimizer_discrim_patchHiddem.zero_grad()
            self.optimizer_discrim_patchRecovery.zero_grad()

            Marked = self.preprocessing_network(Cover)
            # Gauss = self.gaussian(Marked)
            Attacked = self.jpeg_layer(Marked) # 0.5*Marked+0.5*self.jpeg_layer(Marked)

            portion_attack, portion_maxPatch = self.config.attack_portion * (0.75 + 0.25 * self.roundCount),\
                                               self.config.crop_size * (0.75 + 0.25 * self.roundCount)
            Cropped_out, cropout_label, cropout_mask = self.cropout_layer(Attacked,
                                                                          require_attack=portion_attack,max_size=portion_maxPatch)

            Recovered = self.revert_network(Cropped_out, cropout_mask)

            """ Discriminate """
            # d_target_label_cover = torch.full((batch_size, 1), self.cover_label,dtype=torch.float32).cuda()
            # d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label,dtype=torch.float32).cuda()
            # g_target_label_encoded = torch.full((batch_size, 1), self.cover_label,dtype=torch.float32).cuda()
            """Discriminator A"""
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
            loss_D_B = self.backward_D_basic(self.discriminator_patchRecovery, Cover, Recovered)
            loss_D_B.backward()
            self.optimizer_discrim_patchRecovery.step()
            # d_loss_on_cover_B = 0
            # for i in range(5):
            #     d_on_cover = self.discriminator_HiddenRecovery(self.crop_layer(Cover))
            #     d_loss_on_cover_B += self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            # d_loss_on_cover_B.backward()
            # d_loss_on_recovery = 0
            # for i in range(5):
            #     d_on_encoded = self.discriminator_HiddenRecovery(self.crop_layer(Recovered.detach()))
            #     d_loss_on_recovery += self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            # d_loss_on_recovery.backward()
            # # print("-- Adversary B on Cover:{0:.6f},on Recovery:{1:.6f} --".format(d_loss_on_cover_B,d_loss_on_recovery))
            # self.optimizer_discrim_HiddenRecovery.step()
            """Losses"""
            loss_R256_local = self.mse_loss(Recovered * cropout_mask, Cover * cropout_mask)/portion_attack * 100

            loss_cover = self.getVggLoss(Marked, Cover)

            """Adversary Loss"""
            # d_on_encoded_for_enc = self.discriminator_CoverHidden(Marked)
            # g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            g_loss_adv_enc = self.criterionGAN(self.discriminator_patchHidden(Marked), True)
            g_loss_adv_recovery = self.criterionGAN(self.discriminator_patchRecovery(Recovered), True)
            # g_loss_adv_recover = 0
            loss_R256_global = 0
            report_str, max_patch_vgg_loss = '                                     ', 0
            for i in range(8):
                crop_shape = self.crop_layer.get_random_rectangle_inside(Recovered)
                Recovered_portion = self.crop_layer(Recovered, shape=crop_shape)
                Cover_portion = self.crop_layer(Cover, shape=crop_shape)
                # d_on_encoded_for_recovery = self.discriminator_HiddenRecovery(Recovered_portion)
                patch_vggLoss = self.getVggLoss(Recovered_portion, Cover_portion)
                max_patch_vgg_loss = max(patch_vggLoss.item(), max_patch_vgg_loss)
                loss_R256_global += patch_vggLoss
                # g_loss_adv_recovery += self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
                # report_str += "Patch {0:.6f} ".format(patch_vggLoss)
            loss_R256_global  = loss_R256_global * max_patch_vgg_loss / loss_R256_global.item()
            loss_R256 = (loss_R256_local + loss_R256_global) / 2
            loss_enc_dec = self.config.hyper_recovery * loss_R256 + loss_cover * self.config.hyper_cover  # + loss_mask * self.config.hyper_mask
            loss_enc_dec += g_loss_adv_enc * self.config.hyper_discriminator + g_loss_adv_recovery * self.config.hyper_discriminator
            print("Loss backward Begin")
            loss_enc_dec.backward()
            print("Loss backward End")
            self.optimizer_preprocessing_network.step()
            self.optimizer_revert_network.step()

            print(
                "Cropout Local Loss {0:.6f}, Patch Loss {1:.6f}, Total Loss {2:.6f}"
                .format(loss_R256_local,loss_R256_global, loss_R256))
            # print(report_str)


        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': 0, #loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_R256.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }

        return losses, (Marked, Recovered, Cropped_out)

        # """ Localizer (deleted) """
        # x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
        # x_1_gaussian = self.gaussian(x_1_crop)
        # x_1_resize = self.resize_layer(x_1_gaussian)
        # x_1_attack = self.jpeg_layer(x_1_crop)
        # pred_label = self.localizer(x_1_attack.detach())
        # loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
        # loss_localization.backward()
        # self.optimizer_localizer.step()

    def test_on_batch(self, Cover):
        batch_size = Cover.shape[0]
        self.preprocessing_network.eval()
        self.revert_network.eval()
        self.discriminator_CoverHidden.eval()
        self.discriminator_HiddenRecovery.eval()

        with torch.enable_grad():
            """ Run, Train the discriminator"""
            self.optimizer_preprocessing_network.zero_grad()

            self.optimizer_revert_network.zero_grad()
            self.optimizer_discrim_CoverHidden.zero_grad()
            self.optimizer_discrim_HiddenRecovery.zero_grad()
            Marked = self.preprocessing_network(Cover)

            Attacked = Marked #0.5*Marked+0.5*self.jpeg_layer(Marked)
            portion_attack, portion_maxPatch = self.config.attack_portion * (0.7 + 0.0 * self.roundCount), self.config.crop_size * (0.7 + 0.0 * self.roundCount)
            Cropped_out, cropout_label, cropout_mask = self.cropout_layer(Attacked,
                                                                          require_attack=portion_attack,
                                                                          max_size=portion_maxPatch)
            # Extracted = self.reveal_network(Cropped_out)
            Recovered = self.revert_network(Cropped_out, cropout_mask)
            # Recovered = up_256 * self.alpha + out_256 * (1 - self.alpha)

            """ Discriminate """
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)


            """Losses"""
            # loss_R256_global = self.getVggLoss(Recovered, Cover)
            loss_R256_local = self.mse_loss(Recovered * cropout_mask, Cover * cropout_mask)/portion_attack * 100
            loss_R256 = loss_R256_local
            loss_cover = self.getVggLoss(Marked, Cover)
            loss_enc_dec = self.config.hyper_recovery * loss_R256 + loss_cover * self.config.hyper_cover  # + loss_mask * self.config.hyper_mask
            """Adversary Loss"""
            d_on_encoded_for_enc = self.discriminator_CoverHidden(Marked)
            g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            g_loss_adv_recovery = 0
            for i in range(5):
                d_on_encoded_for_recovery = self.discriminator_HiddenRecovery(self.crop_layer(Recovered))
                g_loss_adv_recovery += self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            loss_enc_dec += g_loss_adv_enc * self.config.hyper_discriminator + g_loss_adv_recovery * self.config.hyper_discriminator

            print(
                "Cropout Local Loss {0:.6f}"
                    .format(loss_R256))

            """ Train PrepNetwork and RevertNetwork """
            # pred_again_label = self.localizer(x_1_attack)
            # loss_localization_again = self.bce_with_logits_loss(pred_again_label, cropout_label)
            # if self.config.useVgg == False:
            #     loss_cover = self.mse_loss(Marked, Cover)
            #     loss_recover = self.mse_loss(Recovered, Cover_32)
            #     # loss_mask = self.mse_loss(Recovered*cropout_mask, Cover*cropout_mask)
            # else:
            #     vgg_on_cov = self.vgg_loss(Cover)
            #     vgg_on_enc = self.vgg_loss(Marked)
            #     loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
            #     vgg_on_recovery = self.vgg_loss(Recovered)
            #     loss_recover = self.mse_loss(vgg_on_cov, vgg_on_recovery)
            # d_on_encoded_for_enc = self.discriminator(Marked)
            # g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            # d_on_encoded_for_recovery = self.discriminator(Recovered)
            # g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            """ Total loss for EncoderDecoder """
            # loss_enc_dec = self.config.hyper_recovery * loss_recover + loss_cover * self.config.hyper_cover# + loss_mask * self.config.hyper_mask
            #                # + loss_localization_again * self.config.hyper_localizer\
            #                 # + g_loss_adv_enc * self.config.hyper_discriminator \
            # loss_enc_dec.backward()
            # self.optimizer_preprocessing_network.step()
            # self.optimizer_revert_network.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': 0,  # loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_R256.item(),
            'loss_discriminator_enc': g_loss_adv_enc.item(),
            'loss_discriminator_recovery': g_loss_adv_recovery.item()
        }
        # Test
        # print("Loss Mask: "+str(loss_mask.item() * self.config.hyper_mask))
        return losses, (Marked, Recovered, Cropped_out)

    def save_state_dict_all(self, path):
        torch.save(self.revert_network.state_dict(), path + '_revert_network.pkl')
        print("Successfully Saved: " + path + '_revert_network.pkl')
        torch.save(self.preprocessing_network.state_dict(), path + '_prep_network.pkl')
        print("Successfully Saved: " + path + '_prep_network.pkl')
        torch.save(self.discriminator_patchHidden, path + '_discriminator_patchHidden.pkl')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pkl')
        torch.save(self.discriminator_patchRecovery, path + '_discriminator_patchRecovery.pkl')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pkl')

    # torch.save(self.discriminator.state_dict(), path + '_discriminator_network.pkl')
        # print("Successfully Saved: " + path + '_discriminator_network.pkl')

    def save_model(self,path):
        torch.save(self.revert_network, path + '_revert_network.pth')
        print("Successfully Saved: " + path + '_revert_network.pth')
        torch.save(self.preprocessing_network, path + '_prep_network.pth')
        print("Successfully Saved: " + path + '_prep_network.pth')
        torch.save(self.discriminator_patchHidden, path + '_discriminator_patchHidden.pth')
        print("Successfully Saved: " + path + '_discriminator_patchHidden.pth')
        torch.save(self.discriminator_patchRecovery, path + '_discriminator_patchRecovery.pth')
        print("Successfully Saved: " + path + '_discriminator_patchRecovery.pth')
        # torch.save(self.discriminator, path + '_discriminator_network.pth')
        # print("Successfully Saved: " + path + '_discriminator_network.pth')

    def load_state_dict_all(self,path):
        # self.discriminator.load_state_dict(torch.load(path + '_discriminator_network.pkl'))
        # print("Successfully Loaded: " + path + '_discriminator_network.pkl')
        self.preprocessing_network.load_state_dict(torch.load(path + '_prep_network.pkl'), strict=False)
        print("Successfully Loaded: " + path + '_prep_network.pkl')
        self.revert_network.load_state_dict(torch.load(path + '_revert_network.pkl'), strict=False)
        print("Successfully Loaded: " + path + '_revert_network.pkl')
        # self.hiding_network.load_state_dict(torch.load(path + '_hiding_network.pkl'))
        # print("Successfully Loaded: " + path + '_hiding_network.pkl')
        # self.reveal_network.load_state_dict(torch.load(path + '_reveal_network.pkl'))
        # print("Successfully Loaded: " + path + '_reveal_network.pkl')

    def load_model(self,path):
        # self.discriminator = torch.load(path + '_discriminator_network.pth')
        # print("Successfully Loaded: " + path + '_discriminator_network.pth')
        self.preprocessing_network = torch.load(path + '_prep_network.pth')
        print("Successfully Loaded: " + path + '_prep_network.pth')
        self.revert_network = torch.load(path + '_revert_network.pth')
        print("Successfully Loaded: " + path + '_revert_network.pth')
        # self.discriminator_patchRecovery = torch.load(path + '_discriminator_patchRecovery.pth')
        # print("Successfully Loaded: " + path + '_discriminator_patchRecovery.pth')
        # self.discriminator_patchHidden = torch.load(path + '_discriminator_patchHidden.pth')
        # print("Successfully Loaded: " + path + '_discriminator_patchHidden.pth')


    # def load_state_dict_pretrain(self, path):
    #     # state = torch.load(path)
    #     # load_state = {k: v for k, v in state.items() if k not in state_list}
    #     # print(state.items())
    #     # model_state = model.state_dict()
    #
    #     # model_state.update(load_state)
    #     self.hiding_network.load_state_dict(torch.load(path + '_pretrain_hiding_Epoch N20.pkl'))
    #     self.reveal_network.load_state_dict(torch.load(path + '_pretrain_reveal_Epoch N20.pkl'))
    #     print("Successfully Loaded: "+path)

    # def pretrain_on_batch(self, Cover, Another):
    #     """
    #         预训练：训练Hiding Images in Images论文的结果，也即把Secret图像隐藏到Cover图像中
    #         其中HidingNetwork和ExtractNetwork的训练主要是为了下面的train_on_batch
    #     """
    #     batch_size = Cover.shape[0]
    #     self.pretrain_net.train()
    #
    #     with torch.enable_grad():
    #         """ Run, Train the discriminator"""
    #         # self.optimizer_localizer.zero_grad()
    #         # self.optimizer_discrim.zero_grad()
    #         Marked, Extracted = self.pretrain_net(Cover, Another)
    #         """ Discriminate """
    #         d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
    #         d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
    #         g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
    #         d_on_cover = self.discriminator(Cover)
    #         d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
    #         d_loss_on_cover.backward()
    #         d_on_encoded = self.discriminator(Marked.detach())
    #         d_on_recovered = self.discriminator(Extracted.detach())
    #         d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
    #         d_loss_on_recovered = self.bce_with_logits_loss(d_on_recovered, d_target_label_encoded)
    #         d_loss_on_fake_total = (d_loss_on_encoded + d_loss_on_recovered) / 2
    #         d_loss_on_fake_total.backward()
    #         self.optimizer_discrim.step()
    #
    #         """ Train PrepNetwork and RevertNetwork """
    #         if self.config.useVgg == False:
    #             loss_cover = self.mse_loss(Marked, Cover)
    #             loss_recover = self.mse_loss(Extracted, Another)
    #         else:
    #             vgg_on_cov = self.vgg_loss(Cover)
    #             vgg_on_another = self.vgg_loss(Another)
    #             vgg_on_enc = self.vgg_loss(Marked)
    #             loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
    #             vgg_on_recovery = self.vgg_loss(Extracted)
    #             loss_recover = self.mse_loss(vgg_on_another, vgg_on_recovery)
    #         d_on_encoded_for_enc = self.discriminator(Marked)
    #         g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
    #         d_on_encoded_for_recovery = self.discriminator(Extracted)
    #         g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
    #         """ Total loss for EncoderDecoder """
    #         # loss_enc_dec = loss_recover * self.config.hyper_recovery + loss_cover * self.config.hyper_cover
    #         loss_enc_dec = g_loss_adv_recovery * self.config.hyper_discriminator + loss_recover * self.config.hyper_recovery \
    #                        + loss_cover * self.config.hyper_cover + g_loss_adv_enc * self.config.hyper_discriminator
    #         # + loss_cover * self.config.hyper_cover\
    #         # + loss_localization_again * self.config.hyper_localizer\
    #         # + g_loss_adv_enc * self.config.hyper_discriminator \
    #         loss_enc_dec.backward()
    #         self.optimizer_encoder_decoder_network.step()
    #
    #     losses = {
    #         'loss_sum': loss_enc_dec.item(),
    #         'loss_localization': 0,  # loss_localization.item(),
    #         'loss_cover': loss_cover.item(),
    #         'loss_recover': loss_recover.item(),
    #         'loss_discriminator_enc': g_loss_adv_enc.item(),
    #         'loss_discriminator_recovery': g_loss_adv_recovery.item()
    #     }
    #     return losses, (Marked, Extracted)