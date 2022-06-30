import os
import time
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import argparse
import pdb

from model import Models
from utils import *
from dataset import VideoDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 4.0) 

def parse_args():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument("--train_path", type=str, default='dataset/charades/Charades_i3d_train_data.h5')
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
    parser.add_argument("--manualSeed", type=int, help='manual seed')
    parser.add_argument("--bs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_init_lr", type=float, default=-1)
    parser.add_argument("--warmup_updates", type=float, default=4000)
    parser.add_argument("--Tmax", type=int, default=10)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lambda_0", type=float, default=0.1)
    parser.add_argument("--lambda_1", type=float, default=1.0)
    parser.add_argument("--lambda_2", type=float, default=1.0)
    parser.add_argument("--lambda_3", type=float, default=1.0)
    parser.add_argument("--delta0", type=float, default=0.16)
    parser.add_argument("--delta1", type=float, default=0.05)
    parser.add_argument("--delta2", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=-0.8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.4)
    # save_path
    parser.add_argument("--save_path", type=str, default='model')
    parser.add_argument("--result_path", type=str, default='result/')
    parser.add_argument("--n_save", type=int, default=50)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 100)
    print(args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    
    dataset = VideoDataset(args.train_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                 batch_size=args.bs, drop_last=True, shuffle=True, num_workers=0)
    print('Successfully load the training data')
    main_model = Models(args.v_layer, args.v_hidden, args.s_layer, args.s_hidden, args.l_cross, args.n_loc, args.n_feature, args.n_frame, args.n_hidden, 7, args.n_inter)
    main_model.apply(weights_init)
    optimizer = optim.Adam(main_model.parameters(), lr=args.lr, betas=(0.5, 0.999)) 
    main_model = main_model.cuda()
    num_batches = len(dataloader)
   
    smoothloss = torch.nn.SmoothL1Loss().cuda()


    sent = Variable(torch.FloatTensor(args.bs, args.n_word, 300)).cuda()
    rgb = Variable(torch.FloatTensor(args.bs, args.n_frame, 1024)).cuda()
    flow = Variable(torch.FloatTensor(args.bs, args.n_frame, 1024)).cuda()
    gt_loc = Variable(torch.FloatTensor(args.bs, 2)).cuda()
    length = Variable(torch.FloatTensor(args.bs)).cuda()
    loc = Variable(torch.zeros(args.bs, 2).float()).cuda()
    v_id = Variable(torch.IntTensor(args.bs)).cuda()
    sen_mask = Variable(torch.IntTensor(args.bs, args.n_word)).cuda()
    video_mask = Variable(torch.IntTensor(args.bs, args.n_frame)).cuda()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
   
    ave_start_policy_loss_all = []
    ave_start_value_loss_all = []
    ave_end_policy_loss_all = []
    ave_end_value_loss_all = []
    ave_loc_loss_all = []
    ave_iou_loss_all = []
    ave_dis_loss_all = []
       
    all_run = 1
    
    print('Start training!!!')
    main_model.train()

    for epoch in range(args.max_epoch):
        data_iter = iter(dataloader)

        iteration = 0
        start_policy_loss_epoch = []
        start_value_loss_epoch = []
        end_policy_loss_epoch = []
        end_value_loss_epoch = []
        loc_loss_epoch = []
        iou_loss_epoch = []
        dis_loss_epoch = []

        while iteration < num_batches:
            start_entropies = torch.zeros(args.Tmax, args.bs).cuda()
            start_values = torch.zeros(args.Tmax, args.bs).cuda()
            start_log_probs = torch.zeros(args.Tmax, args.bs).cuda()
            start_rewards = torch.zeros(args.Tmax, args.bs).cuda()
            end_entropies = torch.zeros(args.Tmax, args.bs).cuda()
            end_values = torch.zeros(args.Tmax, args.bs).cuda()
            end_log_probs = torch.zeros(args.Tmax, args.bs).cuda()
            end_rewards = torch.zeros(args.Tmax, args.bs).cuda()
            Previous_IoUs = torch.zeros(args.Tmax, args.bs).cuda()
            Predict_IoUs = torch.zeros(args.Tmax, args.bs).cuda()
            Previous_dis = torch.zeros(args.Tmax, args.bs, 2).cuda()
            Predict_dis = torch.zeros(args.Tmax, args.bs, 2).cuda()
            locations = torch.zeros(args.Tmax, args.bs, 2).cuda()
            start_masks = torch.zeros(args.Tmax, args.bs).cuda()
            start_mask = Variable(torch.ones(args.bs).float()).cuda()
            end_masks = torch.zeros(args.Tmax, args.bs).cuda()
            end_mask = Variable(torch.ones(args.bs).float()).cuda()
            start_h_t = Variable(torch.zeros(args.bs, args.n_hidden).float()).cuda()
            end_h_t = Variable(torch.zeros(args.bs, args.n_hidden).float()).cuda()
            
            # load data
            data = data_iter.next()
            _1, _2, _3, _4, _5, _6, _7, _8 = data
            sent.copy_(_1)
            rgb.copy_(_2)
            flow.copy_(_3)
            length.copy_(_4)
            gt_loc.copy_(_5)
            v_id.copy_(_6)
            sen_mask.copy_(_7)
            video_mask.copy_(_8)
            
            # normalize the ground truth
            gt_loc[:, 0] =  gt_loc[:, 0] / length
            gt_loc[:, 1] = gt_loc[:, 1] / length


            # converge the visual feature from rgb and optical flow
            gv = torch.cat((rgb, flow), 2).detach()
            cur_time = time.time()
            for step in range(args.Tmax):

                # initialize the temporal location
                if step == 0:
                    loc[:, 0] = 1 / 4.0 * torch.ones_like(loc[:, 0])
                    loc[:, 1] = 3 / 4.0 * torch.ones_like(loc[:, 1])
                    visual_fea = None
                    sen_fea = None 
                else:
                    for i in range(args.bs):
                        if start_action[i] == 6:
                            start_mask[i] = 0
                        if end_action[i] == 6:
                            end_mask[i] = 0

                pre_location = loc.clone()
                P_start = loc[:, 0].clone()
                P_end = loc[:, 1].clone()
                
                # model
                visual_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, \
                       end_logit, end_v_t, p_tIoU, p_loc, p_dis = main_model(step, gv, \
                              sent.detach(), visual_fea, sen_fea, pre_location.detach(), \
                              sen_mask.detach(), video_mask.detach(),\
                              start_h_t.detach(), end_h_t.detach())

                # transformations
                start_prob = F.softmax(start_logit, dim=1)
                start_log_prob = F.log_softmax(start_logit, dim=1)
                start_entropy = -(start_log_prob * start_prob).sum(1)
                start_entropies[step, :] = start_entropy

                start_action = start_prob.multinomial(num_samples=1).data
                start_log_prob = start_log_prob.gather(1, start_action)
                start_action = start_action.cpu().numpy()[:, 0]

                end_prob = F.softmax(end_logit, dim=1)
                end_log_prob = F.log_softmax(end_logit, dim=1)
                end_entropy = -(end_log_prob * end_prob).sum(1)
                end_entropies[step, :] = end_entropy

                end_action = end_prob.multinomial(num_samples=1).data
                end_log_prob = end_log_prob.gather(1, end_action)
                end_action = end_action.cpu().numpy()[:, 0]


                Predict_IoUs[step, :] = p_tIoU.squeeze(1)
                locations[step, :, :] = p_loc
                gl = gt_loc.expand(args.bs, 2)
                Previous_dis[step, :, :] = gl - loc
                Predict_dis[step, :, :] = p_dis
                 

                if step == 0:
                    Previous_IoU = calculate_RL_IoU_batch(loc, gt_loc)
                else:
                    Previous_IoU = current_IoU

                Previous_IoUs[step, :] = Previous_IoU
                loc = renew_state(loc, start_action, end_action, start_mask, end_mask, \
                                  args.delta0, args.delta1, args.delta2)
                current_IoU = calculate_RL_IoU_batch(loc, gt_loc)
                C_start = loc[:, 0].clone()
                C_end = loc[:, 1].clone()
                start_reward = calculate_reward_batch_withstop_start(P_start, C_start, C_end, gt_loc, \
                                                args.beta, args.threshold, start_action, args.gamma)
                start_values[step, :] = start_v_t.squeeze(1)
                start_log_probs[step, :] = start_log_prob.squeeze(1)
                start_rewards[step, :] = start_reward
                start_masks[step, :] = start_mask

                end_reward = calculate_reward_batch_withstop_end(P_end, C_end, gt_loc, C_start, \
                                                args.beta, args.threshold, end_action, args.gamma)
                end_values[step, :] = end_v_t.squeeze(1)
                end_log_probs[step, :] = end_log_prob.squeeze(1)
                end_rewards[step, :] = end_reward
                end_masks[step, :] = end_mask
           
            per_time = (time.time() - cur_time) / float(args.bs)
            # compute losses
            start_value_loss = 0
            end_policy_loss = 0
            end_value_loss = 0
            start_policy_loss = 0
            s_dis_loss = 0
            e_dis_loss = 0
            loc_loss = 0
            iou_loss = 0
            idx = 0

            mask_1 = torch.zeros_like(Previous_IoUs)
            
            
            for j in range(args.bs):
                mask_start = start_masks[:, j]
                mask_end = end_masks[:, j]
                n_s = 0
                n_e = 0
                
                # mark the processed experiences
                for index in range(args.Tmax):
                    if mask_start[index] == 1:
                        n_s = n_s + 1
                    if mask_end[index] == 1:
                        n_e = n_e + 1
         
                num = max(n_s, n_e)
              
                for k in reversed(range(num)):
                          
                    sign_s = 0
                    sign_e = 0

                    if k == n_s - 1:
                        S_R = args.gamma * start_values[k][j] + start_rewards[k][j]
                        sign_s = 1
                    if k == n_e - 1:
                        E_R = args.gamma * end_values[k][j] + end_rewards[k][j]
                        sign_e = 1
                    if k < n_s - 1:
                        S_R = args.gamma * S_R + start_rewards[k][j]
                        sign_s = 1
                    if k < n_e - 1:
                        E_R = args.gamma * E_R + end_rewards[k][j]
                        sign_e = 1
                    
                    if sign_s == 1:
                        s_advantage = S_R.detach() - start_values[k][j]
                        start_value_loss = start_value_loss + s_advantage.pow(2)
                        start_policy_loss = start_policy_loss - start_log_probs[k][j] * s_advantage.detach() - args.lambda_0 * start_entropies[k][j]
                    if sign_e == 1:
                        e_advantage = E_R.detach() - end_values[k][j]
                        end_value_loss = end_value_loss + e_advantage.pow(2)
                        end_policy_loss = end_policy_loss - end_log_probs[k][j] * e_advantage.detach() - args.lambda_0 * end_entropies[k][j]

                    iou_loss += torch.abs(Previous_IoUs[k, j] - Predict_IoUs[k, j])
                    mask_1[k, j] = Previous_IoUs[k, j] > 0.4
                    idx += 1
           
            start_policy_loss /= idx
            start_value_loss /= idx
            end_policy_loss /= idx
            end_value_loss /= idx
            iou_loss /= idx

            loc_id = 0
            for i in range(len(mask_1)):
                for j in range(len(mask_1[i])):
                    if mask_1[i, j] == 1:
                        loc_loss += (torch.abs(gt_loc[j][0].detach() - locations[i][j][0]) +
                                     torch.abs(gt_loc[j][1].detach() - locations[i][j][1])) / 2.0
                 
                        s_dis_loss += smoothloss(Predict_dis[i][j][0], Previous_dis[i][j][0].detach())
                        e_dis_loss += smoothloss(Predict_dis[i][j][1], Previous_dis[i][j][1].detach())
                        loc_id += 1

            dis_loss = s_dis_loss + e_dis_loss    
            if loc_id != 0:
                loc_loss /= loc_id
                dis_loss /= loc_id             
          
            start_policy_loss_epoch.append(float(start_policy_loss))
            start_value_loss_epoch.append(float(start_value_loss))
            end_policy_loss_epoch.append(float(end_policy_loss))
            end_value_loss_epoch.append(float(end_value_loss))
            loc_loss_epoch.append(float(loc_loss))
            iou_loss_epoch.append(float(iou_loss))   
            dis_loss_epoch.append(float(dis_loss))     
            policy_loss = start_policy_loss + end_policy_loss
            value_loss = start_value_loss + end_value_loss
            loss = policy_loss + args.lambda_1 * value_loss + args.lambda_3 *(iou_loss + args.lambda_2 * loc_loss) + dis_loss
            optimizer.zero_grad()
            adjust_lr(optimizer, args.lr, args.warmup_init_lr, args.warmup_updates, all_run)
            #loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            optimizer.step()


            if all_run % args.n_save == 0:
                torch.save(main_model.state_dict(), '%s/model_%d.pth' % (args.save_path, all_run))
                print('save the %d model' % iteration)
            print("Flops is %f s" % (per_time))
            print("Train Epoch: %d | Index: %d | policy loss: %f" % (epoch, iteration+1, policy_loss))            
            print("Train Epoch: %d | Index: %d | value loss: %f" % (epoch, iteration+1, value_loss)) 
            print("Train Epoch: %d | Index: %d | IoU loss: %f" % (epoch, iteration+1, iou_loss)) 
            print("Train Epoch: %d | Index: %d | Location loss: %f" % (epoch, iteration+1, loc_loss))
            print("Train Epoch: %d | Index: %d | Distance loss: %f" % (epoch, iteration+1, dis_loss)) 
          

            iteration = iteration + 1
            all_run += 1

        ave_start_policy_loss = sum(start_policy_loss_epoch) / len(start_policy_loss_epoch)
        ave_start_policy_loss_all.append(ave_start_policy_loss)
        print("Average Start Agent Policy Loss for Train Epoch %d : %f" % (epoch, ave_start_policy_loss))

        ave_start_value_loss = sum(start_value_loss_epoch) / len(start_value_loss_epoch)
        ave_start_value_loss_all.append(ave_start_value_loss)
        print("Average Start Agent Value Loss for Train Epoch %d : %f" % (epoch, ave_start_value_loss))


        ave_end_policy_loss = sum(end_policy_loss_epoch) / len(end_policy_loss_epoch)
        ave_end_policy_loss_all.append(ave_end_policy_loss)
        print("Average End Agent Policy Loss for Train Epoch %d : %f" % (epoch, ave_end_policy_loss))

        ave_end_value_loss = sum(end_value_loss_epoch) / len(end_value_loss_epoch)
        ave_end_value_loss_all.append(ave_end_value_loss)
        print("Average End Agent Value Loss for Train Epoch %d : %f" % (epoch, ave_end_value_loss))


        ave_iou_loss = sum(iou_loss_epoch) / len(iou_loss_epoch)
        ave_iou_loss_all.append(ave_iou_loss)
        print("Average IoU Loss for Train Epoch %d : %f" % (epoch, ave_iou_loss))

        ave_loc_loss = sum(loc_loss_epoch) / len(loc_loss_epoch)
        ave_loc_loss_all.append(ave_loc_loss)
        print("Average Location Loss for Train Epoch %d : %f" % (epoch, ave_loc_loss))
   
        ave_dis_loss = sum(dis_loss_epoch) / len(dis_loss_epoch)
        ave_dis_loss_all.append(ave_dis_loss)
        print("Average Location Loss for Train Epoch %d : %f" % (epoch, ave_dis_loss))

    result_path = args.result_path

    with open(result_path + "iteration_ave_start_policy_loss.pkl", "wb") as f_file:
        pickle.dump(ave_start_policy_loss_all, f_file)
    x = np.arange(1, len(ave_start_policy_loss_all) + 1)
    plt.figure(1)
    plt.plot(x, ave_start_policy_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Start Agent Polcy Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_start_policy_loss.png")
    plt.close(1)

    with open(result_path + "iteration_ave_start_value_loss.pkl", "wb") as f_file:
        pickle.dump(ave_start_value_loss_all, f_file)
    plt.figure(1)
    plt.plot(x, ave_start_value_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Start Agent Value Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_start_value_loss.png")
    plt.close(1)


    with open(result_path + "iteration_ave_end_policy_loss.pkl", "wb") as f_file:
        pickle.dump(ave_end_policy_loss_all, f_file)
    x = np.arange(1, len(ave_end_policy_loss_all) + 1)
    plt.figure(1)
    plt.plot(x, ave_end_policy_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average End Agent Polcy Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_end_policy_loss.png")
    plt.close(1)

    with open(result_path + "iteration_ave_end_value_loss.pkl", "wb") as f_file:
        pickle.dump(ave_end_value_loss_all, f_file)
    plt.figure(1)
    plt.plot(x, ave_end_value_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average End Agent Value Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_end_value_loss.png")
    plt.close(1)

    with open(result_path + "iteration_ave_iou_loss.pkl", "wb") as f_file:
        pickle.dump(ave_iou_loss_all, f_file)
    plt.figure(1)
    plt.plot(x, ave_iou_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average IoU Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_iou_loss.png")
    plt.close(1)
    
    with open(result_path + "iteration_ave_loc_loss.pkl", "wb") as f_file:
        pickle.dump(ave_loc_loss_all, f_file)
    plt.figure(1)
    plt.plot(x, ave_loc_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Location Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_loc_loss.png")
    plt.close(1)

    with open(result_path + "iteration_ave_dis_loss.pkl", "wb") as f_file:
        pickle.dump(ave_dis_loss_all, f_file)
    plt.figure(1)
    plt.plot(x, ave_dis_loss_all, "r-")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.title("Average Distance Loss iteration")
    plt.xticks(fontsize=8)
    plt.savefig(result_path + "iteration_ave_dis_loss.png")
    plt.close(1)
        

