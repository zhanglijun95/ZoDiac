from PIL import Image
import cv2
import torch
import os
from imwatermark import WatermarkEncoder, WatermarkDecoder
from torchvision import transforms


class Watermarker:
    def encode(self, img_path, output_path, prompt=''):
        raise NotImplementedError

    def decode(self, img_path):
        raise NotImplementedError


class InvisibleWatermarker(Watermarker):
    def __init__(self, wm_text, method):
        if method == 'rivaGan':
            WatermarkEncoder.loadModel()
        self.method = method
        self.encoder = WatermarkEncoder()
        self.wm_type = 'bytes'
        self.wm_text = wm_text
        self.decoder = WatermarkDecoder(self.wm_type, len(self.wm_text) * 8)

    def encode(self, img_path, output_path):
        img = cv2.imread(img_path)
        self.encoder.set_watermark(self.wm_type, self.wm_text.encode('utf-8'))
        out = self.encoder.encode(img, self.method)
        cv2.imwrite(output_path, out)

    def decode(self, img_path):
        wm_img = cv2.imread(img_path)
        wm_text_decode = self.decoder.decode(wm_img, self.method)
        return wm_text_decode