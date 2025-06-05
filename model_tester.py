
from email.headerregistry import DateHeader
import math
import os
import random

import numpy as np
import torch
from dataset import time_stamped_audio
from dataset_preprossesor import pre_process_data_set
from model.model import transcription_model

import timed_tokenizer

SAVE_CHECKPOINT_DIR = "model_checkpoints"
BEST_MODEL_PATH = os.path.join(SAVE_CHECKPOINT_DIR, "best_model.pt")


model,_=transcription_model.load(BEST_MODEL_PATH)

model.eval()

model.to("cuda")

val_dataset =  time_stamped_audio("train")



while True:


    index=random.randint(0,len(val_dataset)-1)
    print(f"index: {index}")

    src_mask,memory_mask,raw_tokens,audio,clip_len,can_be_end=pre_process_data_set(val_dataset[index])
    
    
    audio=torch.from_numpy(audio).to("cuda").to(torch.float32)
    audio=audio.unsqueeze(0)
    audio=audio.unsqueeze(0)


    text=torch.Tensor(raw_tokens[:-1]).to("cuda").to(torch.long)
    text=text.unsqueeze(0)

    output_text=torch.Tensor(raw_tokens[1:]).to("cuda").to(torch.long)

    can_be_end=can_be_end.to("cuda").to(torch.bool)
    can_be_end=can_be_end.unsqueeze(0)

    

    output=model.forward(
        audio=audio,
        text=text,
        tgt_mask_tensor=src_mask.to("cuda"),
        audio_memory_mask=memory_mask.to("cuda"),
        can_be_end=can_be_end
    )
    output=output.squeeze(1)
    print(f"output: {output}")
    print(f"output: {output.shape}")

    output=torch.nn.Softmax(dim=1)(output)

    print(f"output: {output}")
    print(f"output: {output.shape}")
    
    print(f"text: {output_text.shape}")

    output2=torch.gather(output, dim=1, index=output_text.unsqueeze(1) ) 

    print(f"output: {output2}")
    print(f"output: {output2.shape}")

    for i in range(output2.shape[0]):
        print(f"prob: {round((float(abs(math.log(output2[i])))),4)} word: {timed_tokenizer.audio_tokenizer.decode([raw_tokens[1+i]])}")

    input("")
