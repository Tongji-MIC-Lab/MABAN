import os
import sys
from gensim.models import KeyedVectors
import pickle
import time
import numpy as np
import pdb
from nltk import word_tokenize

# load the annotaion files of Charades-STA
f1 = open("/data/sun/Video_Retrieval/Charades/charades_sta_test.txt", "r") 
f2 = open("/data/sun/Video_Retrieval/Charades/charades_sta_train.txt", "r")

# the file path of pretrained glove embedding
word2vec_output_file = 'glove.840B.300d.word2vec.txt'
start_time = time.time()
word2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
end_time = time.time()
print('load model cost', end_time - start_time, 's')

max_words = 10

def load_data(in_f, out_f):
    clip_sentvec = []
    for line in in_f:
        v_data = line.split("##", 1)
        v_info = v_data[0].split()
        v_name = v_info[0]
    
        v_sen = v_data[1].split("\n", 1)[0].lower()
        loc = np.zeros([2])
        loc[0] = float(v_info[1])
        loc[1] = float(v_info[2])
            
        sen_mask = np.zeros((max_words)) 
        sen_feature = np.zeros((max_words, 300)) 
        tokens = word_tokenize(v_sen)
        for i in range(len(tokens)):
            if i == max_words:
                break
            sen_mask[i] = 1
            token = tokens[i]
            try:
                token_vec = word2vec[token]
            except KeyError:
                token_vec = word2vec['unk']
            sen_feature[i] = token_vec
        clip_sentvec.append([v_name, loc, sen_feature, sen_mask])
   
    with open(out_f, 'wb+') as f:
        pickle.dump(clip_sentvec, f, pickle.HIGHEST_PROTOCOL)

load_data(f1, 'charades/test_clip_sentvec_charades.pkl')
load_data(f2, 'charades/train_clip_sentvec_charades.pkl')

f1.close()
f2.close()
