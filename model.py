import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb



class Observation(nn.Module):
    def __init__(self, v_layer, v_hidden, s_layer, s_hidden, l_cross, n_loc, n_feature, n_frame):
        super(Observation, self).__init__()
        n_v = 1024 * 2
        n_s = 300
        self.v_hidden = v_hidden
        self.s_hidden = s_hidden 
        self.v_layer = v_layer
        self.s_layer = s_layer
        self.v_rnn = nn.GRU(n_v, v_hidden, v_layer, batch_first=True, bidirectional=True, dropout=0.5)
        self.s_rnn = nn.GRU(n_s, s_hidden, s_layer, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(v_hidden * 2, l_cross)
        self.fc2 = nn.Linear(s_hidden * 2, l_cross)
        self.fc3 = nn.Linear(2, n_loc)
        n = 2 * (v_hidden + s_hidden) + l_cross + n_loc
        self.fc4 = nn.Linear(n, n_feature)
        self.n_frame = n_frame

    def forward(self, step, gv, sen, visual_fea, sen_fea, loc, sen_mask, video_mask):
        
        bs = len(sen)
        if step == 0:
            v_len = video_mask.sum(1)
            sort_lens, idx_sort = torch.sort(v_len, descending=True)
            _, idx_unsort = torch.sort(idx_sort, descending=False)
            sort_gv = gv[idx_sort]
            pack_gv = pack_padded_sequence(sort_gv, sort_lens.cpu().numpy(), batch_first=True)

            v_hid = torch.zeros(2 * self.v_layer, bs, self.v_hidden).cuda()

            visual_fea, _ = self.v_rnn(pack_gv, v_hid)
            visual_fea = pad_packed_sequence(visual_fea, batch_first=True, total_length=gv.size(1))[0]
            visual_fea = visual_fea[idx_unsort]


            lens = sen_mask.sum(1)
            sort_lens, idx_sort = torch.sort(lens, descending=True)
            _, idx_unsort = torch.sort(idx_sort, descending=False)
        
            sort_sen = sen[idx_sort]
            pack_sen = pack_padded_sequence(sort_sen, sort_lens.cpu().numpy(), batch_first=True)

            s_hid = torch.zeros(2 * self.s_layer, bs, self.s_hidden).cuda()
        
            words, _ = self.s_rnn(pack_sen, s_hid)
            words = pad_packed_sequence(words, batch_first=True, total_length=sen.size(1))[0]
            words = words[idx_unsort]
       
            sen_fea = torch.zeros_like(words[:, 0, :]).cuda() 
            for i in range(bs):
                sen_fea[i] = words[i, int(lens[i] - 1)]

        v_len = video_mask.sum(1)
        gv_fea = torch.zeros(bs, visual_fea.size(2)).cuda()
        lv_fea = torch.zeros(bs, visual_fea.size(2)).cuda()
        for i in range(bs):
            gv_fea[i] = torch.mean(visual_fea[i, :int(v_len[i])], dim=0)
            start = int(loc[i, 0] * (int(v_len[i]) - 1))
            end = int(loc[i, 1] * (int(v_len[i]) - 1))
            if start <= end:
                lv_fea[i, :] = torch.mean(visual_fea[i, start:end+1, :], dim=0)
        
        vs = F.relu(self.fc1(lv_fea) * self.fc2(sen_fea))
        loc_fea = F.relu(self.fc3(loc))
        out = torch.cat((sen_fea, gv_fea, vs, loc_fea), dim=1)
        feature = F.relu(self.fc4(out))

        return visual_fea, sen_fea, feature

class policy_network(nn.Module):
    def __init__(self, n_hid, n_input, n_action, n_inter):
        super(policy_network, self).__init__()
        self.fc1 = nn.Linear(n_input, n_inter)
        self.fc2 = nn.Linear(n_inter, 1)
        self.rnn = nn.GRUCell(n_hid, n_hid)
        self.i2h = nn.Linear(n_input + 1, n_hid)
        self.h2h = nn.Linear(n_hid, n_hid)
        self.fc_pi = nn.Linear(n_hid, n_action)

    def forward(self, g_t, h_t_prev):
        # temporal distance regression
        p_dis = self.fc2(F.relu(self.fc1(g_t)))
        inputs = torch.cat((p_dis, g_t), dim=1)
        # policy branch
        h1 = self.i2h(inputs)
        h2 = self.h2h(h_t_prev)
        h_t = self.rnn(h1 + h2, h_t_prev)
        del h1
        del h2
        logit = self.fc_pi(h_t)
        return h_t, logit, p_dis

class critic_network(nn.Module):
    def __init__(self, n_input, n_output):
        super(critic_network, self).__init__()
        self.fc = nn.Linear(n_input, n_output)
    def forward(self, h_t):
        v_t = self.fc(h_t.detach())
        return v_t

class multi_task(nn.Module):
    def __init__(self, n_input):
        super(multi_task, self).__init__()
        self.fc1 = nn.Linear(n_input, 1)
        self.fc2 = nn.Linear(n_input, 2)
       
    def forward(self, feature):
        #out = self.fc(feature)
        p_tIoU = self.fc1(feature)
        p_loc = self.fc2(feature) 
        return p_tIoU, p_loc

class Models(nn.Module):
    def __init__(self, v_layer, v_hidden, s_layer, s_hidden, l_cross, n_loc, n_feature, n_frame, n_hid, n_action, n_inter):
        super(Models, self).__init__()
        self.obs = Observation(v_layer, v_hidden, s_layer, s_hidden, l_cross, n_loc, n_feature, n_frame)
        self.start_pn = policy_network(n_hid, n_feature, n_action, n_inter) 
        self.start_cn = critic_network(n_hid, 1)
        self.end_pn = policy_network(n_hid, n_feature, n_action, n_inter) 
        self.end_cn = critic_network(n_hid, 1)
        self.mt = multi_task(n_feature)
    def forward(self, step, gv, sen, visual_fea, sen_fea, loc, sen_mask, video_mask, start_h_t_prev, end_h_t_prev):
        gv_fea, sen_fea, feature = self.obs(step, gv, sen, visual_fea, sen_fea, loc, sen_mask, video_mask)
        start_h_t, start_logit, p_s = self.start_pn(feature, start_h_t_prev)
        start_v_t = self.start_cn(start_h_t)
        end_h_t, end_logit, p_e = self.end_pn(feature, end_h_t_prev)
        end_v_t = self.end_cn(end_h_t)
        p_tIoU, p_loc = self.mt(feature)
        p_dis = torch.cat((p_s, p_e), dim = 1)
        del feature
        return gv_fea, sen_fea, start_h_t, start_logit, start_v_t, end_h_t, end_logit, end_v_t, p_tIoU, p_loc, p_dis



