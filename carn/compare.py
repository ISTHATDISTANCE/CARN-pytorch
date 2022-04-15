import lpips
import os
import torch
from torchvision import transforms
import skimage.metrics as metrics
from PIL import Image
import numpy as np

loss_fn_alex = lpips.LPIPS(net='alex')

datasets = [(5, 'Set5'), (14, 'Set14'), (100, 'B100')]

models = os.listdir('sample')
print(models)
# models = ['mycarn', 'mycarn_1000', 'carn_201000', 'denseBlock_89000']

convert_tensor = transforms.ToTensor()

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    if im1.shape[0] != 3:
        im1 = np.broadcast_to(im1, (3, im1.shape[1], im1.shape[2]))
    psnr = metrics.peak_signal_noise_ratio(im1, im2, data_range=1)
    return psnr

def main():
    lpips_score = np.zeros(len(models))
    psnr_score = np.zeros(len(models))
    for idx, model in enumerate(models):
        for size, dataset in datasets:
            for scale in [4]:
                for i in range(1, size+1):
                    img_HR = Image.open('dataset/%s/x%d/img_%.3d_SRF_%d_HR.png' % (dataset, scale, i, scale))
                    img_SR = Image.open('sample/%s/%s/x%s/SR/img_%.3d_SRF_%d_SR.png' % (model, dataset, scale, i, scale))
                    tensor = convert_tensor(img_HR)
                    tensor1 = convert_tensor(img_SR)
                    d1 = loss_fn_alex(tensor, tensor1)
                    psnr_s = psnr(tensor.numpy(), tensor1.numpy())
                    lpips_score[idx] = lpips_score[idx] + d1
                    psnr_score[idx] = psnr_score[idx] + psnr_s
    print(lpips_score)
    print(psnr_score)
    print("lpips ranking:")
    for i in np.argsort(lpips_score):
        print('\t', models[i])
    print("psnr ranking:")
    for i in np.argsort(psnr_score)[::-1]:
        print('\t', models[i])

if __name__ == '__main__':
    main()