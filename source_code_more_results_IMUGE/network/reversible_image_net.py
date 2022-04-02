# %matplotlib inline
import torch
import torch.nn as nn

from DeepSteg import HidingNetwork
# from encoder.encoder_decoder import EncoderDecoder
from config import GlobalConfig
from decoder.revert_unet import Revert_Unet
from decoder.revert import Revert
from decoder.revert_P3 import RevertNew_P3
from decoder.revertRerun256 import Revert
from encoder.prep_unet import PrepNetwork_Unet
from loss.vgg_loss import VGGLoss
from network.reveal import RevealNetwork
from noise_layers.cropout import Cropout
from noise_layers.dropout import Dropout
from noise_layers.jpeg_compression import JpegCompression
from decoder.revert_sameAsPrep import RevertNew_P7
from discriminator.discriminator import Discriminator
from network.pure_upsample import PureUpsampling
from decoder.revert_ying import Revert_ying

class ReversibleImageNetwork_ying:
    def __init__(self, username, config=GlobalConfig()):
        super(ReversibleImageNetwork_ying, self).__init__()
        """ Settings """
        self.alpha = 1.0
        self.config = config
        self.device = self.config.device
        self.username = username
        """ Generator Network"""
        #self.pretrain_net = Pretrain_deepsteg(config=config).to(self.device)
        # self.encoder_decoder = Net(config=config).to(self.device)
        self.preprocessing_network = PrepNetwork_Unet(input_channel=3, config=config).to(self.device)
        # self.hiding_network = HidingNetwork().to(self.device)
        # self.reveal_network = RevealNetwork().to(self.device)
        """ Recovery Network """
        self.revert_network_P7 = RevertNew_P7(input_channel=3, config=config).to(self.device)
        self.revert_network_P5 = RevertNew_P5(input_channel=3, config=config).to(self.device)
        self.revert_network_P3 = RevertNew_P3(input_channel=3, config=config).to(self.device)
        # self.revert_network = Revert(config=config).to(self.device)
        """Localize Network"""
        # if self.username=="qichao":
        #     self.localizer = LocalizeNetwork(config).to(self.device)
        # else:
        #     self.localizer = LocalizeNetwork_noPool(config).to(self.device)
        """Discriminator"""
        # self.discriminator = Discriminator(config).to(self.device)
        # self.cover_label = 1
        # self.encoded_label = 0
        """Vgg"""

        self.vgg_loss = VGGLoss(3, 1, False)
        self.vgg_loss.to(self.device)

        """Loss"""
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss = nn.MSELoss().to(self.device)

        """Optimizer"""
        # self.optimizer_hiding_network = torch.optim.Adam(self.hiding_network.parameters())
        self.optimizer_preprocessing_network = torch.optim.Adam(self.preprocessing_network.parameters())
        self.optimizer_revert_P7_network = torch.optim.Adam(self.revert_network_P7.parameters())
        self.optimizer_revert_P5_network = torch.optim.Adam(self.revert_network_P5.parameters())
        self.optimizer_revert_P3_network = torch.optim.Adam(self.revert_network_P3.parameters())
        # self.optimizer_reveal_network = torch.optim.Adam(self.reveal_network.parameters())
        # self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        """Attack Layers"""
        self.cropout_layer = Cropout(config).to(self.device)
        self.jpeg_layer = JpegCompression(self.device).to(self.device)
        # self.resize_layer = Resize(config, (0.5, 0.7)).to(self.device)
        # self.gaussian = Gaussian(config).to(self.device)
        # self.dropout_layer = Dropout(config,(0.4,0.6)).to(self.device)

    def train_on_batch(self, Cover):
        """
            训练方法：先额外训练单个Secret图像向Cover图像嵌入和提取的网络（可以通过读取预训练结果），
            然后将Secret图像送入PrepNetwork(基于Unet)做处理（置乱），送进嵌入和提取网络，
            提取得到的图像再送进RevertNetwork得到近似原图（B），再填充到原图中
            Loss：B与原图的loss，Hidden与原图的loss
        """
        batch_size = Cover.shape[0]
        self.preprocessing_network.train()
        self.revert_network_P7.train()
        self.revert_network_P5.train()
        self.revert_network_P3.train()
        # self.discriminator.train()
        self.alpha -= 1 / 20480
        if self.alpha < 0.0:
            self.alpha = 0.0

        with torch.enable_grad():
            """ Run, Train the discriminator"""
            self.optimizer_preprocessing_network.zero_grad()
            # self.optimizer_hiding_network.zero_grad()
            # self.optimizer_reveal_network.zero_grad()
            self.optimizer_revert_P7_network.zero_grad()
            self.optimizer_revert_P5_network.zero_grad()
            self.optimizer_revert_P3_network.zero_grad()
            # self.optimizer_discrim.zero_grad()
            Marked = self.preprocessing_network(Cover)
            Attacked = self.jpeg_layer(Marked)
            Cropped_out, cropout_label, cropout_mask = self.cropout_layer(Attacked)
            # Extracted = self.reveal_network(Cropped_out)

            p7_final = self.revert_network_P7(Cropped_out)
            p5_final = self.revert_network_P5(Cropped_out.detach(), p7_final.detach())
            # p3_final = self.revert_network_P3(p5_final)
            """ First Train P5 fully, then train alpha*P7+(1-alpha)*P5"""
            loss_fromP7_global = self.mse_loss(p7_final, Marked.detach())
            loss_fromP7_local = self.mse_loss(p7_final * cropout_mask, Marked.detach() * cropout_mask) * 5
            loss_fromP7 = loss_fromP7_local + loss_fromP7_global
            loss_fromP5_global = self.mse_loss(p5_final, Marked.detach())
            loss_fromP5_local = self.mse_loss(p5_final * cropout_mask, Marked.detach() * cropout_mask) * 5
            loss_fromP5 = loss_fromP5_local + loss_fromP5_global
            loss_fromP5.backward()
            self.optimizer_revert_P5_network.step()
            p5_final_again = self.revert_network_P5(Cropped_out, p7_final)
            Recovered = p7_final*self.alpha + p5_final_again*(1-self.alpha)
            loss_cover = self.mse_loss(Marked, Cover)
            loss_recover_global = self.mse_loss(Recovered, Marked.detach())
            loss_recover_local = self.mse_loss(Recovered * cropout_mask, Marked.detach() * cropout_mask) * 5
            loss_recover = loss_recover_local + loss_recover_global
            loss_enc_dec = self.config.hyper_recovery * loss_recover + loss_cover * self.config.hyper_cover  # + loss_mask * self.config.hyper_mask
            loss_enc_dec.backward()
            self.optimizer_preprocessing_network.step()
            self.optimizer_revert_P7_network.step()
            print("Curr alpha: {0:.6f}, (Total) Overall Loss: {1:.6f}, local: {2:.6f},  (P7) Overall Loss: {3:.6f}, local: {4:.6f}, (P5) Overall Loss: {5:.6f}, local : {6:.6f}"
                  .format(self.alpha, loss_recover, loss_recover_local, loss_fromP7, loss_fromP7_local, loss_fromP5, loss_fromP5_local))
            """ Discriminate """
            # d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            # d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            # g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)
            # d_on_cover = self.discriminator(Cover)
            # d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover)
            # d_loss_on_cover.backward()
            # d_on_encoded = self.discriminator(Marked.detach())
            # # d_on_recovered = self.discriminator(Recovered.detach())
            # d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded)
            # # d_loss_on_recovered = self.bce_with_logits_loss(d_on_recovered, d_target_label_encoded)
            #
            # d_loss_on_encoded.backward()
            # self.optimizer_discrim.step()
            """ Train PrepNetwork and RevertNetwork """
            # pred_again_label = self.localizer(x_1_attack)
            # loss_localization_again = self.bce_with_logits_loss(pred_again_label, cropout_label)
            # if self.config.useVgg == False:
            #     loss_cover = self.mse_loss(Marked, Cover)
            #     loss_recover = self.mse_loss(Recovered, Marked.detach())*5
            #     # loss_recover = self.mse_loss(Recovered*cropout_mask, Cover*cropout_mask)*5
            # else:
            #     vgg_on_cov = self.vgg_loss(Cover)
            #     vgg_on_enc = self.vgg_loss(Marked)
            #     loss_cover = self.mse_loss(vgg_on_cov, vgg_on_enc)
            #     vgg_on_recovery = self.vgg_loss(Recovered)
            #     loss_recover = self.mse_loss(vgg_on_cov, vgg_on_recovery)
            # # d_on_encoded_for_enc = self.discriminator(Marked)
            # # g_loss_adv_enc = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded)
            # # d_on_encoded_for_recovery = self.discriminator(Recovered)
            # # g_loss_adv_recovery = self.bce_with_logits_loss(d_on_encoded_for_recovery, g_target_label_encoded)
            # """ Total loss for EncoderDecoder """
            # loss_enc_dec = self.config.hyper_recovery * loss_recover + loss_cover * self.config.hyper_cover# + loss_mask * self.config.hyper_mask
            #                # + loss_localization_again * self.config.hyper_localizer\
            #                 # + g_loss_adv_enc * self.config.hyper_discriminator \
            # loss_enc_dec.backward()
            # self.optimizer_preprocessing_network.step()
            # self.optimizer_revert_P7_network.step()
            # self.optimizer_revert_P3_network.step()

        losses = {
            'loss_sum': loss_enc_dec.item(),
            'loss_localization': 0, #loss_localization.item(),
            'loss_cover': loss_cover.item(),
            'loss_recover': loss_recover.item(),
            'loss_discriminator_enc': 0,#g_loss_adv_enc.item(),
            'loss_discriminator_recovery': 0 # g_loss_adv_recovery.item()
        }
        # Test
        # print("Loss Mask: "+str(loss_mask.item() * self.config.hyper_mask))
        return losses, (Marked, Recovered, Cropped_out, p7_final, p5_final) # Recovered*cropout_mask

        # """ Localizer (deleted) """
        # x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
        # x_1_gaussian = self.gaussian(x_1_crop)
        # x_1_resize = self.resize_layer(x_1_gaussian)
        # x_1_attack = self.jpeg_layer(x_1_crop)
        # pred_label = self.localizer(x_1_attack.detach())
        # loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
        # loss_localization.backward()
        # self.optimizer_localizer.step()

    def validate_on_batch(self, Cover, Another):
        pass
        # batch_size = Cover.shape[0]
        # self.encoder_decoder.eval()
        # self.localizer.eval()
        # with torch.enable_grad():
        #     x_hidden, x_recover, mask, self.jpeg_layer.__class__.__name__ = self.encoder_decoder(Cover, Another)
        #
        #     x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
        #     x_1_gaussian = self.gaussian(x_1_crop)
        #     x_1_attack = self.jpeg_layer(x_1_gaussian)
        #     pred_label = self.localizer(x_1_attack.detach())
        #     loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
        #
        #     loss_cover = self.mse_loss(x_hidden, Cover)
        #     loss_recover = self.mse_loss(x_recover.mul(mask), Cover.mul(mask)) / self.config.min_required_block_portion
        #     loss_enc_dec = loss_localization * self.hyper[0] + loss_cover * self.hyper[1] + loss_recover * \
        #                    self.hyper[2]
        #
        # losses = {
        #     'loss_sum': loss_enc_dec.item(),
        #     'loss_localization': loss_localization.item(),
        #     'loss_cover': loss_cover.item(),
        #     'loss_recover': loss_recover.item()
        # }
        # return losses, (x_hidden, x_recover.mul(mask) + Cover.mul(1 - mask), pred_label, cropout_label)

    def save_state_dict_all(self, path):
        torch.save(self.revert_network_P7.state_dict(), path + '_revert_network_P7.pkl')
        print("Successfully Saved: " + path + '_revert_network_P7.pkl')
        torch.save(self.revert_network_P5.state_dict(), path + '_revert_network_P5.pkl')
        print("Successfully Saved: " + path + '_revert_network_P5.pkl')
        torch.save(self.preprocessing_network.state_dict(), path + '_prep_network.pkl')
        print("Successfully Saved: " + path + '_prep_network.pkl')
        # torch.save(self.hiding_network.state_dict(), path + '_hiding_network.pkl')
        # print("Successfully Saved: " + path + '_hiding_network.pkl')
        # torch.save(self.reveal_network.state_dict(), path + '_reveal_network.pkl')
        # print("Successfully Saved: " + path + '_reveal_network.pkl')
        # torch.save(self.discriminator.state_dict(), path + '_discriminator_network.pkl')
        # print("Successfully Saved: " + path + '_discriminator_network.pkl')

    def save_model(self,path):
        torch.save(self.revert_network_P7, path + '_revert_network_P7.pth')
        print("Successfully Saved: " + path + '_revert_network_P7.pth')
        torch.save(self.revert_network_P5, path + '_revert_network_P5.pth')
        print("Successfully Saved: " + path + '_revert_network_P5.pth')
        torch.save(self.preprocessing_network, path + '_prep_network.pth')
        print("Successfully Saved: " + path + '_prep_network.pth')
        # torch.save(self.hiding_network, path + '_hiding_network.pth')
        # print("Successfully Saved: " + path + '_hiding_network.pth')
        # torch.save(self.reveal_network, path + '_reveal_network.pth')
        # print("Successfully Saved: " + path + '_reveal_network.pth')
        # torch.save(self.discriminator, path + '_discriminator_network.pth')
        # print("Successfully Saved: " + path + '_discriminator_network.pth')

    def load_state_dict_all(self,path):
        # self.discriminator.load_state_dict(torch.load(path + '_discriminator_network.pkl'))
        # print("Successfully Loaded: " + path + '_discriminator_network.pkl')
        self.preprocessing_network.load_state_dict(torch.load(path + '_prep_network.pkl'))
        print("Successfully Loaded: " + path + '_prep_network.pkl')
        self.revert_network_P7.load_state_dict(torch.load(path + '_revert_network_P7.pkl'))
        print("Successfully Loaded: " + path + '_revert_network_P7.pkl')
        self.revert_network_P5.load_state_dict(torch.load(path + '_revert_network_P5.pkl'))
        print("Successfully Loaded: " + path + '_revert_network_P5.pkl')
        # self.hiding_network.load_state_dict(torch.load(path + '_hiding_network.pkl'))
        # print("Successfully Loaded: " + path + '_hiding_network.pkl')
        # self.reveal_network.load_state_dict(torch.load(path + '_reveal_network.pkl'))
        # print("Successfully Loaded: " + path + '_reveal_network.pkl')

    def load_model(self,path):
        # self.discriminator = torch.load(path + '_discriminator_network.pth')
        # print("Successfully Loaded: " + path + '_discriminator_network.pth')
        self.preprocessing_network = torch.load(path + '_prep_network.pth')
        print("Successfully Loaded: " + path + '_prep_network.pth')
        self.revert_network_P7 = torch.load(path + '_revert_network_P7.pth')
        print("Successfully Loaded: " + path + '_revert_network_P7.pth')
        self.revert_network_P5 = torch.load(path + '_revert_network_P5.pth')
        print("Successfully Loaded: " + path + '_revert_network_P5.pth')
        # self.hiding_network = torch.load(path + '_hiding_network.pth')
        # print("Successfully Loaded: " + path + '_hiding_network.pth')
        # self.reveal_network = torch.load(path + '_reveal_network.pth')
        # print("Successfully Loaded: " + path + '_reveal_network.pth')


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