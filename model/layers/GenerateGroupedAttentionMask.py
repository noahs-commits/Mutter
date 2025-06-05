
import torch

import timed_tokenizer


backward_memory_mask_view=1
forward_memory_mask_view=5

class grouped_attention_mask_generator:
    
    """generates the src and memory masks for the grouped attention model"""
    """feed in a token and it will generate a row of both masks"""
    def __init__(self, word_len, audio_length):
        self.word_count=word_len
        self.audio_length=audio_length

        self.audio_start=0
        self.audio_end=0

        self.word_start=0
        self.word_end=0

    def add_token(self,new_token):
        #update start and end of the word and audio group
        self.add_tokens_memory_mask(new_token)
        self.add_tokens_src_mask(new_token)


        #inialised mask with all values set to True representing that all tokens can't see each other
        src_mask_row=torch.full((self.word_count,), True, dtype=torch.bool)
        memory_mask_row=torch.full((self.audio_length,), True, dtype=torch.bool)


        #set the values of the src and memory mask to False for the tokens that can see each other
        src_mask_row[self.word_start:self.word_end]=False
        memory_mask_row[self.audio_start-backward_memory_mask_view:self.audio_end+forward_memory_mask_view]=False

        return src_mask_row,memory_mask_row
    def add_tokens_memory_mask(self,new_token):
        match new_token:
            case timed_tokenizer.clear_buffer_id:
                #self.audio_end+=1
                self.audio_start=self.audio_end
            case timed_tokenizer.add_to_buffer_id:
                self.audio_end+=1
            case _:
                pass
    def add_tokens_src_mask(self,new_token):
        match new_token:
            case timed_tokenizer.clear_buffer_id:
                self.word_start=self.word_end
            case _:
                pass
        self.word_end+=1
                
"""
def get_grouped_mask(timed_words):

    # Add one for the potential start token (adjust if your start token logic differs)
    token_count = 1 + sum(tw.token_end - tw.token_start for tw in timed_words)
    # Calculate total time length
    clip_len = sum(tw.time_end - tw.time_start for tw in timed_words)

    # Initialize masks with True
    # interaly we are using the convesion that True means it is masked out and False means it is not masked out
    # however we invert the mask at the end so it uses the standard convention
    # this is done so that we can use the torch.triu function to mask out the upper triangular part of the matrix
    src_mask = torch.full((token_count, token_count), True, dtype=torch.bool)
    memory_mask = torch.full((token_count, clip_len), True, dtype=torch.bool)
  
    for tw in timed_words:

        #generate slices
        
        #all words in the same word can see each other and the last token in the previous word group
        src_mask_slice=[slice(tw.token_start+1, tw.token_end+1),slice(tw.token_start, tw.token_end+1)]
        
        #all tokens in the same word group can see all the audio grouped with it
        memory_mask_slice=[slice(tw.token_start, tw.token_end),slice(tw.time_start, tw.time_end+forward_view_size)]


        src_mask[src_mask_slice]=False

        #we not just marking that everything can see eachother becuase the start of the word group contains control tokens that effect mask size
        #if we don't do this then will have issues becuase we are telling the model how long the word is
        memory_mask[memory_mask_slice]=memory_mask[memory_mask_slice].triu(diagonal=forward_view_size)

        if tw.token_start>0:
            memory_mask[tw.token_start:tw.token_end,tw.time_start-1].fill_(0)

    
    # Set all numbers in the upper triangular part of the matrix to -torch.inf
    src_mask -= torch.triu(torch.full_like(src_mask, torch.inf), diagonal=1)
    
    return src_mask,memory_mask
def get_grouped_src_mask():
    pass
"""