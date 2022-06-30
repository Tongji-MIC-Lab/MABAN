#!/bin/bash


mkdir charades

# change the folder where initial videos locate
cd /data/sun/Video_Retrieval/Charades/Charades_v1/
# get the video duration by ffmpeg
for file in $(ls *)
do
  var=$(ffmpeg -i $file 2>&1 | grep 'Duration'| cut -d '' -f 4 | sed s/,//)
  min=$(expr substr "$var" 16 2)
  sec=$(expr substr "$var" 19 5)
  echo $file $min $sec >> /data/sun/Video_Retrieval/MABAN/dataset/charades/duration.txt
done

# change the folder where MABAN locates
cd /data/sun/Video_Retrieval/MABAN/dataset/

# transform the time respresentation
python video_length.py

# precess the annotation to valiad format and extract text features
# modify the f1, f2 and word2vec_output_file to fit your corresponding files
python cs_construct_clip_sentvec.py

# sample the I3D features and generate the final processed data.
python cs_data_process_i3d.py 
