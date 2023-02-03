import os
import tqdm
import numpy as np
root = "data"
total = 0

prev_path = []
cur_path = []
next_path = []

video_idx = []

interval = 10

for folder_index in tqdm.tqdm(range(88)):
    if folder_index == 65:
        continue
    folder_path = os.path.join(root, "dir-{}".format(folder_index))
    imgs = list(os.listdir(folder_path))
    imgs = sorted([int(_[:-4]) for _ in imgs])
    length = len(imgs)
    total += length
    start  = imgs[0]
    end = imgs[-1]
    tmp_path = []
    for i in range(length//interval):
        if i < 3:
            continue
        img_name = "{}.jpg".format(i*interval + start)
        tmp_path.append(os.path.join("dir-{}".format(folder_index),img_name))
    prev_path += tmp_path[:-2]
    cur_path += tmp_path[1:-1]
    next_path += tmp_path[2:]

    video_i = folder_index if folder_index < 65 else folder_index-1
    video_idx += [video_i] * len(tmp_path[:-1])

ytb_meta = {"cur_path":cur_path, "next_path":next_path, "prev_path":prev_path, "video_idx":video_idx}

num_samples = len(cur_path)
num_train_samples = int(len(cur_path)*0.9)
ytb_meta_train = {"cur_path":cur_path[:num_train_samples], "next_path":next_path[:num_train_samples], "prev_path":prev_path[:num_train_samples], "video_idx":video_idx[:num_train_samples]}
ytb_meta_val = {"cur_path":cur_path[num_train_samples:], "next_path":next_path[num_train_samples:], "prev_path":prev_path[num_train_samples:], "video_idx":video_idx[num_train_samples:]}
np.save("ytb_meta_trip", ytb_meta, allow_pickle=True)
np.save("ytb_meta_train_trip", ytb_meta_train, allow_pickle=True)
np.save("ytb_meta_val_trip", ytb_meta_val, allow_pickle=True)
print(total, len(cur_path))
