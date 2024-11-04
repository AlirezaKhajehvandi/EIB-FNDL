# Example run: python -W ignore BRISQUE.py --test_dir /root/autodl-tmp/Result/RetinexNet/ExDark --read_subfolder True

import os
from glob import glob
from PIL import Image
import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque
import argparse
from tqdm import tqdm
# from pybrisque import BRISQUE


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir', type=str,
                        default="./data/images/", # put the images file correctly
                        help='directory for testing inputs')
    parser.add_argument('--read_subfolder', type=bool, default=False)                        
    args = parser.parse_args()
    return args
# default="./data_metric/our/DICM/",
def cal_loe(inp):
    pass


def cal_brisque(inp, i, AssertionError_count):
    imgOri = Image.open(inp)
    img = img_as_float(imgOri)
    # Convert the image to grayscale if it's not already
    if len(img.shape) == 3:  # Check if it's an RGB image
        from skimage.color import rgb2gray
        np_img = rgb2gray(img)  # Convert to grayscale
    # img = img_as_float(imgOri, force_copy=False)

    try:
        score = brisque.score(np_img)
    except AssertionError:
        score = 0
        AssertionError_count += 1
        print(i, " path[i] is", inp)
    return score, AssertionError_count


def main(args):
    if args.read_subfolder:
        path = glob(os.path.join(args.test_dir, '*/*'))
    else:
        path = glob(os.path.join(args.test_dir, '*'))

    list_brisque = []

    AssertionError_count = 0
    for i in tqdm(range(len(path))):
    # for i in tqdm(range(1550, len(path))):

        # calculate scores
        # print(i, " path[i] is", path[i])
        score, AssertionError_count = cal_brisque(path[i], i, AssertionError_count)
        if score != 0:
        # append to list
            list_brisque.append(score)

    # Average score for the dataset
    print("Have ", AssertionError_count, " times AssertionError.")
    print("Average BRISQUE:", "%.3f" % (np.mean(list_brisque)))



if __name__ == "__main__":
    args = parse_args()
    main(args)


