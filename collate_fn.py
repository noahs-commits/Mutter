import torch
from torch.nn.utils.rnn import pad_sequence
from dataset_preprossesor import pre_process_data_set
from timed_tokenizer import audio_tokenizer
import timed_tokenizer


def collate_fn(batch,head_count,pad_token=timed_tokenizer.ignore_id,max_len=None):

    batch_count=len(batch)

    src_mask_arr,memory_mask_arr,raw_tokens_arr,audio_arr,clip_len_arr,can_be_end_arr=zip(*(pre_process_data_set(item,max_len) for item in batch))

    

    max_clip_len =max(clip_len_arr)
    max_token_len=max(len(raw_tokens) for raw_tokens in raw_tokens_arr)
    max_input_token_len=max_token_len-1

    #create output tensors
    output_audio_tensor       = torch.zeros(batch_count           ,1                  ,max_clip_len*1600,    dtype=torch.float32)
    output_tgt_mask_tensor    = torch.full((batch_count*head_count,max_input_token_len,max_input_token_len),True, dtype=torch.bool)
    output_memory_mask_tensor = torch.full((batch_count*head_count,max_input_token_len,max_clip_len       ),True, dtype=torch.bool)
    
    output_tokens_tensor      = torch.full((batch_count           , max_token_len      ),pad_token ,dtype=torch.long)
    output_can_be_end_tensor  = torch.full((batch_count           , max_input_token_len),pad_token ,dtype=bool)

    for i,(src_mask_tensor,memory_mask_tensor,raw_tokens,audio) in enumerate(zip(*(src_mask_arr,memory_mask_arr,raw_tokens_arr,audio_arr))):
        
        audio_len=len(audio)
        audio_token_length=audio_len//1600
        token_len=len(raw_tokens)
        
        
        #pad the audio tensor to the max clip length
        output_audio_tensor[i,0,:len(audio)]=torch.from_numpy(audio)

        #pad the token tensor to the max token length
        output_tokens_tensor[i,:token_len]=torch.tensor(raw_tokens)


        #repeat the tensor on dimention 0 becuase each head needs its own mask
        src_mask_tensor    = src_mask_tensor   .unsqueeze(0).repeat(head_count,1,1)
        memory_mask_tensor = memory_mask_tensor.unsqueeze(0).repeat(head_count,1,1)

        batch_index = i * head_count

        output_tgt_mask_tensor   [batch_index:batch_index+head_count,:token_len-1,:token_len-1       ]=src_mask_tensor
        output_memory_mask_tensor[batch_index:batch_index+head_count,:token_len-1,:audio_token_length]=memory_mask_tensor

        output_can_be_end_tensor[i,:token_len-1]=can_be_end_arr[i]

    input_tokens =output_tokens_tensor[:,:-1]
    output_tokens=output_tokens_tensor[:,1:]


    return {
        "input_text": input_tokens,
        "output_text": output_tokens,
        "audio": output_audio_tensor,
        "tgt_mask": output_tgt_mask_tensor,
        "memory_mask": output_memory_mask_tensor,
        "can_be_end": output_can_be_end_tensor,
    }
