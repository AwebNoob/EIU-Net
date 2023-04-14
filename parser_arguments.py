import argparse

import losses_new
import network


NETWORK_NAMES = network.__all__
LOSS_NAMES = losses_new.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--device', default='cuda', help='device to run')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='EIU_Net',
                        choices=['EIU_Net', 'CPF_Net', 'AttentionUnet', 'CE_Net', 'ori_UNet', 'DCSAU_Net', 'FAT_Net',
                                 'UNext'])
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=320, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=224, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='Combined_Bce_Dice_Loss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='PH2',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'AdamW'],
                        help='optimizer: ' +
                             ' | '.join(['Adam', 'SGD', 'AdamW']) +
                             ' (default: AdamW)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR',
                                 'CosineAnnealingWarmRestarts'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    # parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=6, type=int)

    config = parser.parse_args()

    return config
