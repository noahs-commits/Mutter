#we time group times into .1 seconds intervals. this represents the indexs of the intevals that contain this word. 
from dataclasses import dataclass

import torch
import audio_preprossesor
import timed_tokenizer
from model.layers.GenerateGroupedAttentionMask import grouped_attention_mask_generator
from timed_tokenizer import audio_tokenizer
from model.layers.GenerateGroupedAttentionMask import forward_memory_mask_view

@dataclass
class index_word():
    tokens: list[int]
    start: int
    end: int
def index_timed_word(time_word):
    return index_word(
        tokens=audio_tokenizer.encode(time_word.word),
        start=int(round(time_word.start*10)),
        end=int(round(time_word.end*10))
    )
@dataclass
class timed_and_index_word():
    tokens: list[int]
    token_start: int
    token_end: int
    time_start: int
    time_end: int

def add_control_tokens(timed_words,clip_len):
    indexed_words=[index_timed_word(tw) for tw in timed_words]

    last_word=0
    padded_indexed_words=[]

    for word in indexed_words:

        for i in range(last_word,word.start):
            padded_indexed_words.append(index_word(tokens=[],start=i,end=i+1))
        last_word=word.end
        padded_indexed_words.append(word)
    for i in range(last_word,clip_len):
        padded_indexed_words.append(index_word(tokens=[],start=i,end=i+1))
    
    output=[]
    token_index=0

    for word in padded_indexed_words:
        new_tokens=[timed_tokenizer.add_to_buffer_id]*(word.end-word.start)+word.tokens+[timed_tokenizer.clear_buffer_id]

        new_token_index=token_index+len(new_tokens)

        output.append(timed_and_index_word(
            tokens=new_tokens,
            token_start=token_index,
            token_end=new_token_index,
            time_start=word.start,
            time_end=word.end
        ))

        token_index=new_token_index
    return output

def pre_process_data_set(item,max_len=None):
    timed_words,sentence,audio=item

    audio,unpadded_len=audio_preprossesor.audio_preprossesor(audio)
    
    clip_len=len(audio)//1600

    output=add_control_tokens(timed_words,unpadded_len)

    raw_tokens=[timed_tokenizer.start_id]+[token for word in output for token in word.tokens]+[timed_tokenizer.end_id]

    assert raw_tokens.count(timed_tokenizer.add_to_buffer_id)==unpadded_len, f"the number of buffer tokens {raw_tokens.count(timed_tokenizer.add_to_buffer_id)} does not match the clip length {clip_len}."


    if max_len is not None and len(raw_tokens)>max_len:
        
        #if the are add to buffer tokens that are trucated then nothing can be the end
        can_be_end_empty=timed_tokenizer.add_to_buffer_id in raw_tokens[max_len:]
        
        # Truncate the raw tokens and audio to the maximum length
        raw_tokens=raw_tokens[:max_len]


        # Truncate the audio to match the length of the raw tokens
        # Calculate the new clip length based on the number of buffer tokens
        clip_len=raw_tokens.count(timed_tokenizer.add_to_buffer_id)+forward_memory_mask_view
        audio=audio[:clip_len*1600]
    else:
        can_be_end_empty=False

    input_tokens_for_transformer=raw_tokens[:-1]

    src_mask_list = []
    memory_mask_list = []

    mask_generator = grouped_attention_mask_generator(
        word_len=len(input_tokens_for_transformer), 
        audio_length=clip_len
    )


    for token_id in input_tokens_for_transformer:
        src_mask_row, memory_mask_row = mask_generator.add_token(token_id)
        src_mask_list.append(src_mask_row)
        memory_mask_list.append(memory_mask_row)
    
    # Stack the lists of rows to create the final 2D mask tensors
    src_mask = torch.stack(src_mask_list)
    memory_mask = torch.stack(memory_mask_list)

    can_be_end = torch.full((len(input_tokens_for_transformer),), False, dtype=bool)

    model_output=torch.tensor(raw_tokens[1:])

    if not can_be_end_empty:
        for i in range(len(model_output)-1,0,-1):
            if model_output[i] == timed_tokenizer.add_to_buffer_id:
                assert timed_tokenizer.end_id not in model_output[:i+1]
                break
            can_be_end[i] = True
    else:
        assert timed_tokenizer.end_id not in raw_tokens


    assert not (torch.eq(model_output,timed_tokenizer.end_id          )&~can_be_end).any(), "Output text contains end_id where can_be_end is False"
    assert not (torch.eq(model_output,timed_tokenizer.add_to_buffer_id)& can_be_end).any(), "Output text contains add_to_buffer_id where can_be_end is True"


    return src_mask,memory_mask,raw_tokens,audio,clip_len,can_be_end
