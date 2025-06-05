import os
import tokenize

import cv2
import numpy as np
import torch
from collate_fn import collate_fn
from dataset import time_stamped_audio
from dataset_preprossesor import pre_process_data_set
from timed_tokenizer import audio_tokenizer

from torch.utils.data import DataLoader, Dataset

"""

data_set=time_stamped_audio("train")

words=data_set[0]

src_mask,memory_mask,raw_tokens,audio,clip_len=pre_process_data_set(words)



mask_src_print = np.array([[0 if value else 1 for value in row] for row in src_mask])
mask_memory_print = np.array([[0 if value else 1 for value in row] for row in memory_mask])


cv2.imwrite('pic1_.png',255*mask_src_print)
cv2.imwrite('pic2_.png',255*mask_memory_print)
#print(data_set[0])"""
def print_mask(mask,path):
    path=os.path.join("pics",path)
    print(path)
    mask_print = np.array([[0 if value else 255 for value in row] for row in mask])
    cv2.imwrite(path, mask_print)
def print_masks(masks,path):

    for i in range(masks.shape[0]):
        sub_mask = masks[i,:,:]
        sub_path = f"{path}\\{i}.png"
        print_mask(sub_mask,sub_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset=time_stamped_audio("train")

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, 2), 
    pin_memory=device == "cuda",
    num_workers=min(4, os.cpu_count() // 2) if device == "cuda" else 0
)

batch = next(iter(train_loader))


print_masks(batch["tgt_mask"],"tgt_mask")
print_masks(batch["memory_mask"],"memory_mask")
#print(batch)

