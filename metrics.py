import numpy as np
import torch
import torch.nn.functional as F

from medpy import metric

'''原本的'''


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    # return iou, dice
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


'''nnUnet metrics'''


def assert_shape(test, reference):
    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""
    if torch.is_tensor(test):
        test = torch.sigmoid(test).data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()
    test = test > 0.5
    reference = reference > 0.5

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""
    # if torch.is_tensor(test):
        # test = torch.sigmoid(test).data.cpu().numpy()
    # if torch.is_tensor(reference):
        # reference = reference.data.cpu().numpy()
    # test = test > 0.5
    # reference = reference > 0.5

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def dice_co(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""
    if torch.is_tensor(test):
        test = torch.sigmoid(test).data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()
    test = test > 0.5
    reference = reference > 0.5

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if torch.is_tensor(test):
        test = torch.sigmoid(test).data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()
    test = test > 0.5
    reference = reference > 0.5

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if torch.is_tensor(test):
        test = torch.sigmoid(test).data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()
    test = test > 0.5
    reference = reference > 0.5

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if torch.is_tensor(test):
        test = torch.sigmoid(test).data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()
    test = test > 0.5
    reference = reference > 0.5

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if torch.is_tensor(test):
        test = torch.sigmoid(test).data.cpu().numpy()
    if torch.is_tensor(reference):
        reference = reference.data.cpu().numpy()
    test = test > 0.5
    reference = reference > 0.5

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


'''C2FTrans'''


def dice_coefficient(pred, gt, smooth=1e-5):
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    if (pred.sum() + gt.sum()) == 0:
        return 1
    # pred_flat = pred.view(N, -1)
    # gt_flat = gt.view(N, -1)
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def sespiou_coefficient2(pred, gt, smooth=1e-5):
    """ computational formula:
        sensitivity = TP/(TP+FN)
        specificity = TN/(FP+TN)
        iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # pred_flat = pred.view(N, -1)
    # gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth) / (TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2 * Precision * Recall / (Recall + Precision + smooth)
    return SE.sum() / N, SP.sum() / N, IOU.sum() / N, Acc.sum() / N, F1.sum() / N, Precision.sum() / N, Recall.sum() / N
