import torch
from torch import nn as nn


def calc_dice(probs, targets, squared=True, classsumm=True, ignore_firstchannel=False):
    """ Computes Dice score

    Args:
        probs (tensor [Batch, Channels, H,W,D]): predicted probabilities (should sum up to 1 over channel dimension)
        target (tensor [Batch, Channels, H,W,D]): targets in one-hot format
        squared (bool) : if true implement squared dice loss (better)
        classum (bool) : whether to sum the loss over the batches
        ignore_firstchannel (bool): whether to include first channel (=background) in summed loss
    Returns:
        list [float] : dice scores of each volume in batch (averaged over the channels)
        return list with dice scores of each image in batch 
    """

    eps = 1e-7
    batch_size = targets.size(0)
    num_classes = targets.size(1)

    probs_flat = probs.view(batch_size, num_classes, -1)
    target_flat = targets.view(batch_size, num_classes, -1)

    assert probs_flat.size() == target_flat.size()

    if squared:
        # For squared implementation square values in nominator before summing 
        Dice = 2. * ((probs_flat * target_flat).sum(2) + eps) / \
            (((probs_flat ** 2)).sum(2) +
                ((target_flat ** 2)).sum(2) + eps)
    else:
        Dice = 2. * ((probs_flat * target_flat).sum(2) + eps) / \
            ((probs_flat).sum(2) +
                (target_flat).sum(2) + eps)

    # Sum dice scores from differentclasses
    if classsumm:
        if ignore_firstchannel is False:
            return Dice.sum(1) / num_classes
        else:
            return Dice[:, 1:, ...].sum(1) / num_classes
    else:
        return Dice


class SoftDiceLoss(nn.Module):
    def __init__(self, squared=True, ignore_firstchannel=False):
        super(SoftDiceLoss, self). __init__()
        self.squared = squared
        self.ignore_firstchannel = ignore_firstchannel

    def forward(self, logits, one_hot_targets):
        """ Calculates Dice loss

        Args:
            logits (tensor[Batch_size x C x H x W x D]): raw predictions from the network
            targets (tensor[Batch_size x C x H x W x D])): binary ground truth. Different classes are stored in different channels as one hot encoding (0/1)
        """

        batch_size = one_hot_targets.size(0)
        probs = torch.softmax(logits, dim=1)

        mean_classDice = calc_dice(
            probs, one_hot_targets,
            squared=self.squared,
            ignore_firstchannel=self.ignore_firstchannel)

        # average all elements in batch
        mean_Dice = mean_classDice.sum() / batch_size
        return 1 - mean_Dice


class DiceAndCE(nn.Module):
    """
    Class to calculate the standard Dice for targets in one-hot format
    """

    def __init__(self, squared=True, ignore_firstchannel=False, CE_weight=1):
        super(DiceAndCE, self). __init__()
        self.CE_weight = CE_weight

        self.dice_loss = SoftDiceLoss(
            squared=squared, ignore_firstchannel=ignore_firstchannel)
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets_onehot):
        """ returns combination of Dice and CE ,loss

        Args:
            logits (tensor[Batch_size x C x H x W x D]): raw network predictions
            targets_onehot (tensor[Batch_size x C x H x W x D]): targets in one_hot format
        Returns:
            float: loss
        """
        # Compute Dice loss
        dice = self.dice_loss(logits, targets_onehot)

        # Compute CE loss (convert targets to long format [B x H x W x D])
        targets_long = torch.argmax(targets_onehot, dim=1).type(torch.long)
        CE = self.CE_loss(logits, targets_long)

        return dice + self.CE_weight * CE
