import json
import re

video = {}
in_file = open('charades/duration.txt', 'r')
out_file = open('charades/charades_length.json', 'w')
while True:
    lines = in_file.readline()
    if not lines:
        break
        pass
    v_info = re.split(r' ', lines)
    v_name = v_info[0][:-4]
    v_min = float(v_info[1])
    v_sec = float(v_info[2])
    v_dur = v_min * 60 + v_sec
    video[v_name] = float(format(v_dur, '.1f'))
    
json.dump(video, out_file)

in_file.close()
out_file.close()
