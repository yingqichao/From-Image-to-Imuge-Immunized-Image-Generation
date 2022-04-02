import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils
from torchvision import utils


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(image.shape[0]):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image

def denormalize_batch(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    image_denorm = torch.empty_like(image)
    image_denorm[:, 0, :, :] = (image[:, 0, :, :].clone() * std[0]) + mean[0]
    image_denorm[:, 1, :, :] = (image[:, 1, :, :].clone() * std[1]) + mean[1]
    image_denorm[:, 2, :, :] = (image[:, 2, :, :].clone() * std[2]) + mean[2]

    # for t in range(image.shape[1]):
    #     image[:, t, :, :] = (image[:, t, :, :] * std[t]) + mean[t]
    return image_denorm

def normalize_batch(image, std, mean):
    ''' normalize a tensor of images.'''

    image_norm = torch.empty_like(image)
    image_norm[:, 0, :, :] = (image[:, 0, :, :].clone()-mean[0]) / std[0]
    image_norm[:, 1, :, :] = (image[:, 1, :, :].clone()-mean[1]) / std[1]
    image_norm[:, 2, :, :] = (image[:, 2, :, :].clone()-mean[2]) / std[2]

    # for t in range(image.shape[1]):
    #     image[:, t, :, :] = (image[:, t, :, :]-mean[t]) / std[t]
    return image_norm

def imshow(input_img, text, std, mean):
    '''Prints out an image given in tensor format.'''
    imgs_tsor = torch.cat(input_img, 0)
    img = utils.make_grid(imgs_tsor)
    img = denormalize(img, std, mean)
    npimg = img.detach().cpu().numpy()
    if img.shape[0] == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(text)
    plt.show()
    return

def plt_plot(hist):
    plt.plot(hist)
    plt.title('hist_loss_localization')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.show()

def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(image, name, folder, std=None, mean=None, resize_to=None):
    # images = original_images[:original_images.shape[0], :, :, :].cpu()
    # watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
    #
    # # scale values to range [0, 1] from original range of [-1, 1]
    # images = (images + 1) / 2
    # watermarked_images = (watermarked_images + 1) / 2
    #
    # imgs_tsor = torch.cat(image, 0)
    #image = image[0]
    if std is not None:
        image = denormalize(image, std, mean)
    if resize_to is not None:
        image = F.interpolate(image, size=resize_to)
    #
    # stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, name)
    torchvision.utils.save_image(image, filename, image.shape[0], normalize=False)
    print('Image saved: '+filename)


# def sorted_nicely(l):
#     """ Sort the given iterable in the way that humans expect."""
#     convert = lambda text: int(text) if text.isdigit() else text
#     alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
#     return sorted(l, key=alphanum_key)
#
#
# def last_checkpoint_from_folder(folder: str):
#     last_file = sorted_nicely(os.listdir(folder))[-1]
#     last_file = os.path.join(folder, last_file)
#     return last_file
#
#
#
#
#
# # def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
# def load_last_checkpoint(checkpoint_folder):
#     """ Load the last checkpoint from the given folder """
#     last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
#     checkpoint = torch.load(last_checkpoint_file)
#
#     return checkpoint, last_checkpoint_file
#
#
# def model_from_checkpoint(hidden_net, checkpoint):
#     """ Restores the hidden_net object from a checkpoint object """
#     hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
#     hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
#     hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
#     hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])
#
#
# def log_progress(losses_accu):
#     log_print_helper(losses_accu, logging.info)
#
#
# def print_progress(losses_accu):
#     log_print_helper(losses_accu, print)
#
#
# def log_print_helper(losses_accu, log_or_print_func):
#     max_len = max([len(loss_name) for loss_name in losses_accu])
#     for loss_name, loss_value in losses_accu.items():
#         log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


# def create_folder_for_run(runs_folder, experiment_name):
#     if not os.path.exists(runs_folder):
#         os.makedirs(runs_folder)
#
#     this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')
#
#     os.makedirs(this_run_folder)
#     os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
#     os.makedirs(os.path.join(this_run_folder, 'images'))
#
#     return this_run_folder


# def write_losses(file_name, losses_accu, epoch, duration):
#     with open(file_name, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if epoch == 1:
#             row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
#             writer.writerow(row_to_write)
#         row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
#             '{:.0f}'.format(duration)]
#         writer.writerow(row_to_write)