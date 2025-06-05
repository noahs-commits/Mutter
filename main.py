print("starting")

import torch

from dataset import time_stamped_audio
from dataset_preprossesor import pre_process_data_set
#from dataset import time_stamped_audio




train_dataset = time_stamped_audio("train")

print("dataset loaded")

src_mask_arr,memory_mask_arr,raw_tokens_arr,audio_arr,clip_len_arr=pre_process_data_set(train_dataset[0])
print("pre_process_data_set")
print(f"{src_mask_arr   =}")
print(f"{memory_mask_arr=}")
print(f"{raw_tokens_arr =}")
print(f"{audio_arr      =}")
print(f"{clip_len_arr   =}")
print("type")
print(f"{type(src_mask_arr   )=}")
print(f"{type(memory_mask_arr)=}")
print(f"{type(raw_tokens_arr )=}")
print(f"{type(audio_arr      )=}")
print(f"{type(clip_len_arr   )=}")
print("size")
print(f"{src_mask_arr.size()   =}")
print(f"{memory_mask_arr.size()=}")
print(f"{len(raw_tokens_arr)   =}")
print(f"{audio_arr.size        =}")
print(f"{audio_arr.dtype       =}")

print(f"{len(audio_arr)=}")