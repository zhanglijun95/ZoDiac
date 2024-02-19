import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import torch
import torch.nn as nn
from .watson_vgg import WatsonDistanceVgg
from .pytorch_ssim import SSIM

class LossProvider(nn.Module):
    def __init__(self, loss_weights: list, device):
        super(LossProvider, self).__init__()
        self.loss_weights = loss_weights

        self.loss_img, self.loss_w = nn.MSELoss(), nn.L1Loss()
        self.loss_ssim = SSIM()

        # add perceptive loss
        loss_percep = WatsonDistanceVgg(reduction='sum')
        loss_percep.load_state_dict(torch.load('./loss/rgb_watson_vgg_trial0.pth', map_location='cpu'))
        loss_percep = loss_percep.to(device)
        self.loss_per = lambda pred_img, gt_img: loss_percep((1+pred_img)/2.0, (1+gt_img)/2.0)/ pred_img.shape[0]

    def __call__(self, pred_img_tensor, gt_img_tensor, init_latents, wm_pipe):
        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        lossW = self.loss_w(init_latents_fft[wm_pipe.watermarking_mask], wm_pipe.gt_patch[wm_pipe.watermarking_mask])*self.loss_weights[3]
        lossI = self.loss_img(pred_img_tensor, gt_img_tensor)*self.loss_weights[0]
        lossP = self.loss_per(pred_img_tensor, gt_img_tensor)*self.loss_weights[1]
        lossS = (1-self.loss_ssim(pred_img_tensor, gt_img_tensor))*self.loss_weights[2]
        loss = lossW + lossI + lossP + lossS
        logging.info(f'Watermark {lossW.item():.4f}, Image {lossI.item():.4f}, Perp {lossP.item():.4f}, SSIM {lossS.item():.4f} Total Loss {loss.item():.4f}')
        return loss
