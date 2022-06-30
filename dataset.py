import torch
import torch.utils.data as data
import numpy as np
import h5py

class VideoDataset(data.Dataset):
    def __init__(self, data_dir):
        
        data = h5py.File(data_dir, 'r')
        self.sents = data['sens']
        self.rgbs = data['rgbs']
        self.flows = data['flows']
        self.lengths = data['lens']
        self.gt_locs = data['locs']
        self.ids = data['ids']
        self.sen_masks = data['sen_masks']
        self.video_masks = data['video_masks']

    def __getitem__(self, index):
        sent = self.sents[index]
        rgb = self.rgbs[index]
        flow = self.flows[index]
        length = self.lengths[index]
        gt_loc = self.gt_locs[index]
        v_id = self.ids[index]
        sen_mask = self.sen_masks[index]
        video_mask = self.video_masks[index]

        return sent, rgb, flow, length, gt_loc, v_id, sen_mask, video_mask
    def __len__(self):
        return len(self.sents)  
        
