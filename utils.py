import torch
import numpy as np
import torch.nn as nn
import pdb
from torch.autograd import Variable



def renew_state(pre_location, a_s, a_e, s_mask, e_mask, delta0, delta1, delta2):
    bs = len(a_s)
    location = pre_location.clone()
    for i in range(bs):
        start = float(pre_location[i, 0])
        end = float(pre_location[i, 1])
        if s_mask[i] == 1:
            if a_s[i] == 0:
                start = start + delta0
            elif a_s[i] == 1:
                start = start + delta1
            elif a_s[i] == 2:
                start = start + delta2
            elif a_s[i] == 3:
                start = start - delta0
            elif a_s[i] == 4:
                start = start - delta1
            elif a_s[i] == 5:
                start = start - delta2

        if e_mask[i] == 1:
            if a_e[i] == 0:
                end = end + delta0
            elif a_e[i] == 1:
                end = end + delta1
            elif a_e[i] == 2:
                end = end + delta2
            elif a_e[i] == 3:
                end = end - delta0
            elif a_e[i] == 4:
                end = end - delta1
            elif a_e[i] == 5:
                end = end - delta2
           
        if start < 0:
            start = 0
        if end > 1:
            end = 1
           
        location[i, 0] = start
        location[i, 1] = end
          
    return location

def renew_state_test(pre_location, a_s, a_e, delta0, delta1, delta2, m_s, m_e):
    location = pre_location.clone()
    start = pre_location[0, 0]
    end = pre_location[0, 1]
    if m_s == 1:
        if a_s == 0:
            start = start + delta0
        elif a_s == 1:
            start = start + delta1
        elif a_s == 2:
            start = start + delta2
        elif a_s == 3:
            start = start - delta0
        elif a_s == 4:
            start = start - delta1
        elif a_s == 5:
            start = start - delta2
    if m_e == 1:
        if a_e == 0:
            end = end + delta0
        elif a_e == 1:
            end = end + delta1
        elif a_e == 2:
            end = end + delta2
        elif a_e == 3:
            end = end - delta0
        elif a_e == 4:
            end = end - delta1
        elif a_e == 5:
            end = end - delta2
   
    if start < 0:
        start = 0
    if end > 1:
        end = 1
    
    location[0, 0] = start
    location[0, 1] = end
 
    return location


def calculate_reward_batch_withstop_start(Previous_start, Current_start, Current_end, gt_loc, beta, threshold, s_a, gamma):
    batch_size = len(Previous_start)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        gt_start = float(gt_loc[i, 0])
        P_dis = abs(Previous_start[i] - gt_start)
        C_dis = abs(Current_start[i] - gt_start)
        if s_a[i] != 6:
            if Current_start[i] >= 0 and Current_start[i] < Current_end[i]:
                if P_dis > C_dis:
                    reward[i] = 1 - C_dis  
                else:
                    reward[i] = 0
            else:
                reward[i] = beta
            reward[i] = float(reward[i]) + P_dis - gamma * C_dis
        else:
            if C_dis <= threshold:
                reward[i] = 1
            else:
                reward[i] = -1
    return reward

def calculate_reward_batch_withstop_end(Previous_end, Current_end, gt_loc, Current_start, beta, threshold, e_a, gamma):
    batch_size = len(Previous_end)
    reward = torch.zeros(batch_size)

    for i in range(batch_size):
        gt_end = float(gt_loc[i, 1])
        P_dis = abs(Previous_end[i] - gt_end)
        C_dis = abs(Current_end[i] - gt_end)
        if e_a[i] != 6:
            if Current_end[i] > Current_start[i] and Current_end[i] <= 1:
                if P_dis > C_dis:
                    reward[i] = 1 - C_dis  
                else:
                    reward[i] = 0
            else:
                reward[i] = beta
            reward[i] = float(reward[i]) + P_dis - gamma * C_dis
        else:
            if C_dis <= threshold:
                reward[i] = 1
            else:
                reward[i] = -1
    return reward



def calculate_RL_IoU_batch(i0, i1):
    # calculate temporal intersection over union
    batch_size = len(i0)
    iou_batch = torch.zeros(batch_size)

    for i in range(len(i0)):
        union = (min(i0[i][0], i1[i][0]), max(i0[i][1], i1[i][1]))
        inter = (max(i0[i][0], i1[i][0]), min(i0[i][1], i1[i][1]))
        # if inter[1] < inter[0]:
        #     iou = 0
        # else:
        iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
        iou_batch[i] = iou
    return iou_batch

def calculate_IoU(i0, i1):
    # calculate temporal intersection over union
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        #nn.init.orthogonal_(m.weight.data, 1.0)
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, lr, warmup_init_lr, warmup_updates, num_updates):
    warmup_end_lr = lr
    if warmup_init_lr < 0:
        warmup_init_lr = warmup_end_lr
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
    decay_factor = warmup_end_lr * warmup_updates**0.5
    if num_updates < warmup_updates:
        lr = warmup_init_lr + num_updates*lr_step
    else:
        lr = decay_factor * num_updates**-0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

