
from email.headerregistry import DateHeader
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from dataset import time_stamped_audio
from dataset_preprossesor import pre_process_data_set
from model.model import transcription_model

import timed_tokenizer

SAVE_CHECKPOINT_DIR = "model_checkpoints"
BEST_MODEL_PATH = os.path.join(SAVE_CHECKPOINT_DIR, "best_model.pt")


model,_=transcription_model.load(BEST_MODEL_PATH)

model.eval()

model.to("cuda")

val_dataset =  time_stamped_audio("test")



while True:


    index=random.randint(0,len(val_dataset)-1)
    print(f"index: {index}")
    
    dataset=val_dataset[index]
    print(f"dataset: {dataset}")
    print

    src_mask,memory_mask,raw_tokens,audio,clip_len,can_be_end=pre_process_data_set(dataset)
    
    
    #audio=np.zeros(len(audio),dtype=np.float32)
    audio=torch.from_numpy(audio).to("cuda").to(torch.float32)
    audio=audio.unsqueeze(0)
    audio=audio.unsqueeze(0)


    text=torch.Tensor(raw_tokens[:-1]).to("cuda").to(torch.long)
    text=text.unsqueeze(0)

    output_text=torch.Tensor(raw_tokens[1:]).to("cuda").to(torch.long)

    can_be_end=can_be_end.to("cuda").to(torch.bool)
    can_be_end=can_be_end.unsqueeze(0)

    
    input_tokens_list=[timed_tokenizer.start_id]
    input_tokens_tensor=torch.full((1,len(raw_tokens)-1),timed_tokenizer.ignore_id).to("cuda").to(torch.long)

    #pad the input tensor to the max token length

    #audio=torch.full(audio.size(),0.0).to("cuda").to(torch.float32)

    for i in range(len(raw_tokens)-1):
        input_tokens_tensor[0,i]=input_tokens_list[-1]

        output=model.forward(
            audio=audio,
            text=input_tokens_tensor,
            tgt_mask_tensor=src_mask.to("cuda"),
            audio_memory_mask=memory_mask.to("cuda"),
            can_be_end=can_be_end
        )

        #remove the batch dimension and extract the output for the current token
        output=output[i,0,:]

        #softmax on the token dimension
        output=nn.functional.softmax(output)

        #find the most likely token
        prob,next_token=torch.max(output,0,keepdim=True)

        token_str=timed_tokenizer.audio_tokenizer.decode([next_token.item()])
        print(f"token: {token_str} prob: {prob.item()}") 

        input_tokens_list.append(next_token.item())
    input()
