# # %matplotlib inline
# import torch
# import torch.nn as nn
# from encoder.encoder_decoder import EncoderDecoder
# from config import GlobalConfig
# from localizer.localizer import LocalizeNetwork
# from localizer.localizer_noPool import LocalizeNetwork_noPool
# from decoder.extract_naive import Extract_naive
# from encoder.hiding_naive import Hiding_naive
# from noise_layers.cropout import Cropout
# from noise_layers.jpeg_compression import JpegCompression
# from noise_layers.resize import Resize
# from noise_layers.gaussian import Gaussian
# from network.discriminator import Discriminator
# from loss.vgg_loss import VGGLoss
# from encoder.prep_unet import PrepNetwork_Unet
# from decoder.revert_unet import Revert_Unet
# from encoder.prep_naive import PrepNetwork_Naive
#
# class Localize_hanson:
#     def __init__(self, config=GlobalConfig()):
#         super(Localize_hanson, self).__init__()
#         self.config = config
#         self.device = self.config.device
#
#         """Localize Network"""
#         self.localizer = LocalizeNetwork(config).to(self.device)
#
#
#         """Loss"""
#         self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(self.device)
#         self.mse_loss = nn.MSELoss().to(self.device)
#
#         """Optimizer"""
#
#         self.optimizer_localizer = torch.optim.Adam(self.localizer.parameters())
#
#         """Attack Layers"""
#         self.cropout_layer = Cropout(config).to(self.device)
#         self.jpeg_layer = JpegCompression(self.device).to(self.device)
#         self.resize_layer = Resize((0.5, 0.7)).to(self.device)
#         self.gaussian = Gaussian(config).to(self.device)
#
#
#     def train_on_batch(self, Marked, Other):
#         """
#             训练Localizer：可以从一张图中获取哪些区域被认为修改为了别的无关图像
#         """
#         batch_size = Marked.shape[0]
#         self.localizer.train()
#
#         with torch.enable_grad():
#             """ Run, Train the discriminator"""
#             #self.optimizer_localizer.zero_grad()
#             self.optimizer_localizer.zero_grad()
#             Marked_gaussian = self.gaussian(Marked)
#             # x_1_resize = self.resize_layer(x_1_gaussian)
#             Marked_attack = self.jpeg_layer(Marked_gaussian)
#             Cropped, crop_Groundtruth, _ = self.cropout_layer(Marked_attack, Other)
#
#             crop_Predicted = self.localizer(Cropped.detach())
#             loss_localization = self.bce_with_logits_loss(crop_Predicted, crop_Groundtruth)
#             loss_localization.backward()
#             self.optimizer_localizer.step()
#
#         losses = {
#             'loss_localization': loss_localization.item(),
#         }
#         return losses, crop_Predicted
#
#     def validate_on_batch(self, Cover, Another):
#         pass
#         # batch_size = Cover.shape[0]
#         # self.encoder_decoder.eval()
#         # self.localizer.eval()
#         # with torch.enable_grad():
#         #     x_hidden, x_recover, mask, self.jpeg_layer.__class__.__name__ = self.encoder_decoder(Cover, Another)
#         #
#         #     x_1_crop, cropout_label, _ = self.cropout_layer(x_hidden, Cover)
#         #     x_1_gaussian = self.gaussian(x_1_crop)
#         #     x_1_attack = self.jpeg_layer(x_1_gaussian)
#         #     pred_label = self.localizer(x_1_attack.detach())
#         #     loss_localization = self.bce_with_logits_loss(pred_label, cropout_label)
#         #
#         #     loss_cover = self.mse_loss(x_hidden, Cover)
#         #     loss_recover = self.mse_loss(x_recover.mul(mask), Cover.mul(mask)) / self.config.min_required_block_portion
#         #     loss_enc_dec = loss_localization * self.hyper[0] + loss_cover * self.hyper[1] + loss_recover * \
#         #                    self.hyper[2]
#         #
#         # losses = {
#         #     'loss_sum': loss_enc_dec.item(),
#         #     'loss_localization': loss_localization.item(),
#         #     'loss_cover': loss_cover.item(),
#         #     'loss_recover': loss_recover.item()
#         # }
#         # return losses, (x_hidden, x_recover.mul(mask) + Cover.mul(1 - mask), pred_label, cropout_label)
#
#     def save_state_dict(self, path):
#         torch.save(self.localizer.state_dict(), path + '_localizer_network.pkl')
#
#     def load_state_dict(self,path):
#         self.localizer.load_state_dict(torch.load(path + '_localizer_network.pkl'))
#
#
#
