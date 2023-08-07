import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from math import exp


# Dice and focal losses taken from (and slightly modified):
# https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py


# Focal-Tversky loss taken from:
# https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch


# Focal-Tversky loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1.0):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        focal_tversky = (1 - tversky) ** gamma

        return focal_tversky


# Generalized soft dice loss
class DiceLoss(nn.Module):
    def __init__(self, p=1, smooth=1, reduction='mean', weight=None):
        super(DiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight).cuda()

    def forward(self, probs, label):
        '''
        args: probs: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, C, H, W)
        '''
        numer = torch.sum((probs * label), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + label.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        # numer = torch.sum(numer, dim=1)
        # denom = torch.sum(denom, dim=1)
        loss = 1 - (2 * numer + self.smooth)/(denom + self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):

        probs = torch.sigmoid(logits)
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


bce_loss_fn = nn.BCEWithLogitsLoss()
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()
foc_loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')  # FocalTverskyLoss()
dic_loss_fn = DiceLoss()
bce_loss_fn = nn.BCELoss()


def loss_fn(E_seq, S_seq, S_seq_true, pred_flag=True, val_flag=False):
    ''' Compute loss for a batch of sequences 
    
    Args:
    -----
    '''

    n_frames = S_seq_true.shape[-1]
    device = S_seq_true.device
    S_seq_true = [S_seq_true[..., t] for t in range(n_frames)]
    time_weight = [1.0 if t > 0 else 0.0 for t in range(n_frames)]
    total_loss = torch.tensor(0.0, device=device)
    for t, (E, S, S_true) in enumerate(zip(E_seq, S_seq, S_seq_true)):

        # Sum of prediction error signals (self-supervised)
        # E[0] is next frame prediction error
        pred_loss = 0.0 if E is None else sum([torch.mean(e) for e in E])
        
        # Segmentation prediction loss (supervised)
        # seg_loss = dic_loss_fn(S, S_true)  # + foc_loss_fn(S, S_true)
        seg_loss = bce_loss_fn(S, S_true.float())

        # Do not account for prediction error in validation mode
        if pred_flag and not val_flag:
            frame_loss = seg_loss + pred_loss
        else:
            frame_loss = seg_loss  # (seg_loss if seg_loss > 0 else 0.0)
        
        # Weight loss differently for each frame
        if (val_flag and t > 5) or (not val_flag and t > 0):
            total_loss = total_loss + frame_loss * time_weight[t]

    return total_loss / n_frames
