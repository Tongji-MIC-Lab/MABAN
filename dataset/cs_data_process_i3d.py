import numpy as np
import os
import pickle
import json
import h5py
import math
import pdb


def data_processing(rgb_path, flow_path, sen_path, length_path, save_path, index_path):
    with open(sen_path, 'rb') as sen_f:
        v_info = pickle.load(sen_f)
    
    with open(length_path, 'r') as len_f:
        v_length = json.load(len_f) 
    v_id = 0
    id2name = {}
    sens = []
    rgbs = []
    flows = []
    lengths = []
    locs = []
    ids = []
    sen_masks = []
    video_masks = []
    # define the maximum video length
    max_frames = 140
    for per_v in v_info:
        v_name = per_v[0]         
        v_loc = per_v[1]
        v_sen = np.array(per_v[2])
        sen_mask = np.array(per_v[3])
        v_len = v_length[v_name]
        
        v_loc[0] = float(v_loc[0])
        v_loc[1] = float(v_loc[1])
    
        if float(v_loc[1]) > v_len:
            #v_loc[1] = v_len
            v_len = v_loc[1]
        if float(v_loc[0]) > v_len:
            continue
        id2name[v_id] = v_name
        v_loc = np.array(v_loc)       

        # video sampling
        if os.path.exists(rgb_path + v_name + '.npy') and os.path.exists(flow_path + v_name + '.npy'):
            rgb_feature = np.squeeze(np.load(rgb_path + v_name + '.npy'))
            flow_feature = np.squeeze(np.load(flow_path + v_name + '.npy'))
            
            num_v_rgb = len(rgb_feature)
            num_v_flow = len(flow_feature)
            num_v = min(num_v_rgb, num_v_flow)
         
            select_rgb_feature = np.zeros((max_frames, 1024))
            select_flow_feature = np.zeros((max_frames, 1024))
           
            v_mask = np.ones((max_frames))
            if num_v < max_frames:
                select_rgb_feature[:num_v, :] = rgb_feature[:num_v, :]
                select_flow_feature[:num_v, :] = flow_feature[:num_v, :]
                v_mask[num_v:] = np.zeros((max_frames - num_v))
            else:
                v_ix = 0
                v_sp = 0
                for i in range(num_v):
                    v_sp += 1
                    cur_num = math.ceil((num_v - v_ix) / float(max_frames))
                    if v_sp  == cur_num:
                        select_rgb_feature[v_ix] = rgb_feature[i]
                        select_flow_feature[v_ix] = flow_feature[i]
                        v_ix += 1
                        v_sp = 0
            sens.append(v_sen)
            rgbs.append(select_rgb_feature)
            flows.append(select_flow_feature)
            lengths.append(v_len)
            locs.append(v_loc)
            ids.append(v_id)
            sen_masks.append(sen_mask)
            video_masks.append(v_mask)
            v_id += 1
            
        else:
            print(v_name)
            continue   
    sen_f.close()
    sens = np.array(sens)
    rgbs = np.array(rgbs)
    flows = np.array(flows)
    lengths = np.array(lengths)
    locs = np.array(locs)
    ids = np.array(ids)
    sen_masks = np.array(sen_masks)
    video_masks = np.array(video_masks)

    f_v = h5py.File(save_path, "w")
    f_v.create_dataset("sens", dtype='float', data=sens)
    f_v.create_dataset("rgbs", dtype='float', data=rgbs)
    f_v.create_dataset("flows", dtype='float', data=flows)
    f_v.create_dataset("lens", dtype='float', data=lengths)
    f_v.create_dataset("locs", dtype='float', data=locs)
    f_v.create_dataset("ids", dtype='int', data=ids)
    f_v.create_dataset("sen_masks", dtype='int', data=sen_masks)
    f_v.create_dataset("video_masks", dtype='int', data=video_masks)
    f_v.close()
 
    with open(index_path, "w") as f_index:
        json.dump(id2name, f_index)

# change the flow_path and rgb_path according to the I3D path.
flow_path = '/data/sun/Video_Retrieval/Charades/Charades_i3d_flow/'
rgb_path = '/data/sun/Video_Retrieval/Charades/Charades_i3d_rgb/'
test_sen_path = 'charades/test_clip_sentvec_charades.pkl'
train_sen_path = 'charades/train_clip_sentvec_charades.pkl'
length_path = 'charades/charades_length.json'
test_save_path = 'charades/Charades_i3d_test_data.h5'
train_save_path = 'charades/Charades_i3d_train_data.h5'
test_index_path = 'charades/charades_test_index.json'
train_index_path = 'charades/charades_train_index.json'

data_processing(rgb_path, flow_path, test_sen_path, length_path, test_save_path, test_index_path)
data_processing(rgb_path, flow_path, train_sen_path, length_path, train_save_path, train_index_path)


