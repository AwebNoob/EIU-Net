import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose

from tqdm import tqdm

import network
from dataset import Dataset
from metrics import iou_score
from functions import AverageMeter
from albumentations import RandomRotate90, Resize
import time
from network import EIU_Net

from torch.utils.data import DataLoader

from metrics import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='ISIC2018_UNet_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models_EIUNet/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])

    model = network.__dict__[config['arch']](config['input_channels'], config['num_classes'])
    model = model.to(device)

    def add_img(path):
        img_ids_s = []
        with open(path, 'r') as f:
            for lines in f.readlines():
                img_id = lines[:-5]
                img_ids_s.append(img_id)
        return img_ids_s

    test_path = 'test.list'
    test_img_ids = add_img(test_path)

    model.load_state_dict(torch.load('models_EIUNet/%s/test_model_1.pth' %
                                     config['name']))

    model.eval()

    test_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join(""),
        mask_dir=os.path.join(""),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True)

    jc_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    se_avg_meter = AverageMeter()
    sp_avg_meter = AverageMeter()
    acc_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()

    count = 0

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('heatmap', config['name'], str(c)), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_dataset)):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            jc, dice, se, sp, acc, pre, rec = jaccard(output, target), \
                                              dice_co(output, target), \
                                              sensitivity(output, target), \
                                              specificity(output, target), \
                                              accuracy(output, target), precision(output, target), recall(output,
                                                                                                          target)

            jc_avg_meter.update(jc, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            se_avg_meter.update(se, input.size(0))
            sp_avg_meter.update(sp, input.size(0))
            acc_avg_meter.update(acc, input.size(0))
            precision_avg_meter.update(pre, input.size(0))
            recall_avg_meter.update(rec, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('heatmap', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('jc: %.4f ' % jc_avg_meter.avg)
    print('Dice: %.4f ' % dice_avg_meter.avg)
    print('se: %.4f ' % se_avg_meter.avg)
    print('sp: %.4f ' % sp_avg_meter.avg)
    print('acc: %.4f ' % acc_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
