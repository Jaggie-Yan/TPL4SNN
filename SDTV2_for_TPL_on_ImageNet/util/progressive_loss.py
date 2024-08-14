import torch
from torch.nn import functional as F

def Progressive_loss(outputs, labels, criterion):
    gt_mask = _get_gt_mask(outputs[0, ...], labels)
    other_mask = _get_other_mask(outputs[0, ...], labels)
    T = outputs.size(0)
    _outputs = outputs.detach()
    Loss_es = 0
    for t in range(T):
        past_time_sum = torch.sum(_outputs[0:t, ...], dim=0)
        pre_time_sum = (past_time_sum + outputs[t, ...]) / (t+1)
        pre_mean_out = F.softmax(pre_time_sum / 1, dim=1)
        loss_target = ((pre_mean_out * gt_mask).sum(dim=1)).mean()
        loss_other = ((pre_mean_out * other_mask).sum(dim=1)).mean()
        Loss_es += criterion(outputs[t, ...], labels) + 2 * ((loss_target / (loss_other / 100)) ** ((T - t) / T))
    Loss_es = Loss_es / T 
    return Loss_es 


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()  
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool() 
    return mask