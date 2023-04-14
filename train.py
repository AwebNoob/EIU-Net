import argparse
import os
import random
from collections import OrderedDict
from glob import glob

import numpy as np
from torch.utils.data import DataLoader

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
# from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler

from tqdm import tqdm
from albumentations import RandomRotate90, Resize, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations import *
import network
import losses_new
from parser_arguments import parse_args
from dataset import Dataset
from metrics import *
from functions import AverageMeter, str2bool

import warnings

warnings.filterwarnings("ignore")

NETWORK_NAMES = network.__all__
LOSS_NAMES = losses_new.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr_list = []


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'jc': AverageMeter(),
                  }

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)
        jc = jaccard(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['jc'].update(jc, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('jc', avg_meters['jc'].avg),

        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    # 查看lr变化
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
    print('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('jc', avg_meters['jc'].avg),
                        ])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'jc': AverageMeter(),
                  'dice': AverageMeter(),
                  'se': AverageMeter(),
                  'sp': AverageMeter(),
                  'acc': AverageMeter(),
                  }

    model.eval()

    with torch.no_grad():  # 减少内存使用

        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)  # 样本
            target = target.to(device)  # 标签

            output = model(input)
            loss = criterion(output, target)

            jc, dice, se, sp, acc = jaccard(output, target), \
                                    dice_co(output, target), \
                                    sensitivity(output, target), \
                                    specificity(output, target), \
                                    accuracy(output, target)

            # iou, dice_s = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['jc'].update(jc, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['se'].update(se, input.size(0))
            avg_meters['sp'].update(sp, input.size(0))
            avg_meters['acc'].update(acc, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('jc', avg_meters['jc'].avg),
                ('dice', avg_meters['dice'].avg),
                ('se', avg_meters['se'].avg),
                ('sp', avg_meters['sp'].avg),
                ('acc', avg_meters['acc'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('jc', avg_meters['jc'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('se', avg_meters['se'].avg),
                        ('sp', avg_meters['sp'].avg),
                        ('acc', avg_meters['acc'].avg),
                        ])


def main():
    config = vars(parse_args())  # 将parse_args()返回为字典对象

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs('models_EIUNet/%s/' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models_EIUNet/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'BCEDiceLoss':
        criterion = losses_new.__dict__[config['loss']]().cuda()
    elif config['loss'] == 'Combined_Bce_Dice_Loss':
        criterion = losses_new.__dict__[config['loss']]().cuda()
    cudnn.benchmark = True

    # create model
    model = NETWORK_NAMES.__dict__[config['arch']](config['input_channels'], config['num_classes'])
    model = model.to(device)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            params, lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'], amsgrad=False)
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config['min_lr'])
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=False, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code

    def add_img(path):
        img_ids_s = []
        with open(path, 'r') as f:
            for lines in f.readlines():
                img_id = lines[:-5]
                img_ids_s.append(img_id)
        return img_ids_s

    train_path = "train.list"
    val_path = "val.list"
    test_path = "test.list"
    train_img_ids = add_img(train_path)
    val_img_ids = add_img(val_path)
    test_img_ids = add_img(test_path)

    train_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        HorizontalFlip(p=0.25),
        RandomBrightnessContrast(p=0.3),
        ShiftScaleRotate(shift_limit=0.0625, p=0.25),
        CoarseDropout(),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(''),
        mask_dir=os.path.join(''),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(''),
        mask_dir=os.path.join(''),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=True)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('jc', []),
        ('val_loss', []),
        ('val_jc', []),
        ('val_dice', []),
        ('val_se', []),
        ('val_sp', []),
        ('val_acc', []),
    ])

    best_net_score = [0]
    trigger = 0

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch + 1, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        # 网络分数
        net_score = val_log['jc'] + val_log['dice']

        if config['scheduler'] == 'CosineAnnealingWarmRestarts':
            scheduler.step()
        elif config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - jc %.4f - val_loss %.4f - val_jc %.4f'
              % (train_log['loss'], train_log['jc'], val_log['loss'], val_log['jc']))
        '''
        print('loss %.4f - jc %.4f '
              % (train_log['loss'], train_log['jc']))
        '''
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['jc'].append(train_log['jc'])

        log['val_loss'].append(val_log['loss'])
        log['val_jc'].append(val_log['jc'])
        log['val_dice'].append(val_log['dice'])
        log['val_se'].append(val_log['se'])
        log['val_sp'].append(val_log['sp'])
        log['val_acc'].append(val_log['acc'])

        pd.DataFrame(log).to_csv('models_EIUNet/%s/log_1.csv' %
                                 config['name'], index=False)

        trigger += 1

        if net_score > max(best_net_score):
            best_net_score.append(net_score)
            torch.save(model.state_dict(), 'models_EIUNet/%s/model_1.pth' %
                       (config['name']))
            print(net_score, best_net_score)
            print("=> saved val best model")

            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
