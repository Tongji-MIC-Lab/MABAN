import os
import time
import random
import torch
import argparse
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pdb


from model import Models
from utils import *
from dataset import VideoDataset

def parse_args():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument("--test_path", type=str, default='dataset/charades/Charades_i3d_test_data.h5')
    # net specifications
    parser.add_argument("--n_word", type=int, default=10, help='charades=10, activitynet=25')
    parser.add_argument("--v_layer", type=int, default=2)
    parser.add_argument("--s_layer", type=int, default=2)
    parser.add_argument("--v_hidden", type=int, default=256)
    parser.add_argument("--s_hidden", type=int, default=256)
    parser.add_argument("--l_cross", type=int, default=512)
    parser.add_argument("--n_frame", type=int, default=140, help='charades=140, activitynet=200')
    parser.add_argument("--n_loc", type=int, default=128)
    parser.add_argument("--n_inter", type=int, default=256)
    parser.add_argument("--n_feature", type=int, default=1024)
    parser.add_argument("--n_hidden", type=int, default=512)
    # learning parameters
    parser.add_argument("--manualSeed", type=int, default=6, help='manual seed')
    parser.add_argument("--Tmax", type=int, default=10)
    parser.add_argument("--delta0", type=float, default=0.16)
    parser.add_argument("--delta1", type=float, default=0.05)
    parser.add_argument("--delta2", type=float, default=0.02)
    # load_path
    parser.add_argument("--load_path", type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 100)
    
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    
    dataset = VideoDataset(args.test_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                 batch_size=1, drop_last=True, shuffle=False, num_workers=0)
    #print('Successfully load the testing data')
    main_model = Models(args.v_layer, args.v_hidden, args.s_layer, args.s_hidden, args.l_cross, args.n_loc, args.n_feature, args.n_frame, args.n_hidden, 7, args.n_inter)
    main_model.load_state_dict(torch.load(args.load_path))
    main_model = main_model.cuda()
    main_model.eval()

    positive_5 = 0
    positive_7 = 0

    num_batches = len(dataloader)
    step = 0


    sent = Variable(torch.FloatTensor(1, args.n_word, 300)).cuda()
    rgb = Variable(torch.FloatTensor(1, args.n_frame, 1024)).cuda()
    flow = Variable(torch.FloatTensor(1, args.n_frame, 1024)).cuda()
    gt_loc = Variable(torch.FloatTensor(1, 2)).cuda()
    loc = Variable(torch.zeros(1, 2).float()).cuda()
    sen_mask = Variable(torch.IntTensor(1, args.n_word)).cuda()
    video_mask = Variable(torch.IntTensor(1, args.n_frame)).cuda()

    data_iter = iter(dataloader)
    while step < (num_batches - 1):
        data = data_iter.next()
        _1, _2, _3, _4, _5, _6, _7, _8 = data
        start_h_t = Variable(torch.zeros(1, args.n_hidden).float()).cuda()
        end_h_t = Variable(torch.zeros(1, args.n_hidden).float()).cuda()
        sent.copy_(_1)
        rgb.copy_(_2)
        flow.copy_(_3)
        length = float(_4)
        gt_loc.copy_(_5)
        v_id = int(_6)
        sen_mask.copy_(_7)
        video_mask.copy_(_8)

        gt_loc[:, 0] = gt_loc[:, 0] / length
        gt_loc[:, 1] = gt_loc[:, 1] / length
        gv = torch.cat((rgb, flow), 2).detach()
        loc[:, 0] = 1 / 4.0
        loc[:, 1] = 3 / 4.0
        m_s = 1
        m_e = 1

        visual_fea = None
        sen_fea = None

        for i_temp in range(args.Tmax):
            pre_location = loc.clone()

            visual_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, end_logit, end_v_t, p_tIoU, p_loc, p_dis = main_model(i_temp, gv, sent, visual_fea, sen_fea, pre_location, sen_mask, video_mask, start_h_t, end_h_t)

            start_prob = F.softmax(start_logit, dim=1)
            start_action = start_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0]

            end_prob = F.softmax(end_logit, dim=1)
            end_action = end_prob.max(1, keepdim=True)[1].data.cpu().numpy()[0, 0] 
          

            loc = renew_state_test(loc, start_action, end_action, \
                                    args.delta0, args.delta1, args.delta2, m_s, m_e)
            
            del pre_location
            
            if start_action == 6:
                m_s = 0
            if end_action == 6:
                m_e = 0
                
            if m_s == 0 and m_e == 0:
                break
            del start_logit
            del end_logit
            del start_v_t
            del end_v_t
            del p_tIoU
            del p_loc
        tIoU = calculate_IoU(gt_loc[0], loc[0])
        if float(tIoU) >= 0.5:
            positive_5 = positive_5 + 1
        if float(tIoU) >= 0.7:
            positive_7 = positive_7 + 1
        step = step + 1

    iou_5 = float(positive_5) / num_batches * 100
    iou_7 = float(positive_7) / num_batches * 100
    print('The accuray when tIoU is higher than 0.5 is %.2f' % (iou_5))
    print('The accuray when tIoU is higher than 0.7 is %.2f' % (iou_7))
