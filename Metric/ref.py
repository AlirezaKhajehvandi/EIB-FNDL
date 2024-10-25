# NOTE: skimage.__version__ == '0.17.1'
# Example run: python ref.py --test_dir1 /root/autodl-tmp/Result/RetinexNet/low --test_dir2 /root/autodl-tmp/Dataset/Clean_Images/low

import os
import numpy as np
from glob import glob
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch
import lpips
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Purpose: convert 
def transform(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def _psnr(tf_img1, tf_img2):
    return compare_psnr(tf_img1, tf_img2)


def _ssim(tf_img1, tf_img2):
    return compare_ssim(tf_img1, tf_img2, multichannel=True, win_size=3) # NOTE: see multichannel=True for RGB images


def _lpips(tf_img1, tf_img2, loss_fn_alex):
    return loss_fn_alex(tf_img1, tf_img2).item()


def main(args):


    # NOTE: add sorted 
    path_real = sorted(glob(os.path.join(args.test_dir1, '*')))
    path_fake = sorted(glob(os.path.join(args.test_dir2, '*')))

    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    list_psnr = []
    list_ssim = []
    list_lpips = []

    for i in tqdm(range(len(path_real))):

        # read images
        # print("==========================>")
        # print("path_real[i]", path_real[i])
        # print("path_fake[i]", path_fake[i])
        img_real = cv2.imread(path_real[i])
        img_fake = cv2.imread(path_fake[i])

        assert img_real.shape == img_fake.shape, "{} mismatch with {}".format(path_real[i], path_fake[i])

        # convert to torch tensor for lpips calculation
        tes_real = transform(img_real).to(device)
        tes_fake = transform(img_fake).to(device)

        # calculate scores
        psnr_num = _psnr(img_real, img_fake)
        ssim_num = _ssim(img_real, img_fake)
        lpips_num = _lpips(tes_real, tes_fake, loss_fn_alex)
      

        # append to list
        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        list_lpips.append(lpips_num)


    # Average score for the dataset
    print("======={}=======>".format(args.test_dir1))
    print("======={}=======>".format(args.test_dir2))
    print("Average PSNR:", "%.3f" % (np.mean(list_psnr)))
    print("Average SSIM:", "%.3f" % (np.mean(list_ssim)))
    print("Average LPIPS:", "%.3f" % (np.mean(list_lpips)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir1', type=str,
                        default='./data/outputs/all_labels/sice_parts/',
                        help='directory for clean images')
    parser.add_argument('--test_dir2', type=str,
                        # default='./data/outputs/Comparision/Zero_Our_results/original_pre/org_pre_sice/1/', # zero dce
                        default='./data/outputs/Comparision/conf-ff/sice-part/5/', # our
                        # default='./data/outputs/Comparision/SCI_results/all_data/low/', # sci
                        # default='./data/outputs/Comparision/SGZ/SICE_Mix/', # sgz
                        # default='./data/outputs/Comparision/low/', # psenet
                        help='directory for enhanced or restored images')
    args = parser.parse_args()
    main(args)
