import librosa
import numpy as np
from model.layers.GenerateGroupedAttentionMask import forward_memory_mask_view

goal_sampling_rate = 16000
chunk_size = .1
chunk_size = int(round(chunk_size*goal_sampling_rate))

extra_padding = chunk_size*forward_memory_mask_view

def audio_preprossesor(audio):

    array=audio["array"]
    sampling_rate=audio["sampling_rate"]

    librosa_output=librosa.resample(array, orig_sr=sampling_rate, target_sr=goal_sampling_rate)

    padding_length=extra_padding+(-len(librosa_output))%chunk_size

    librosa_output=np.concat((librosa_output, np.zeros(padding_length)), axis=0)

    assert len(librosa_output)%chunk_size==0, "The length of the audio is not a multiple of the chunk size."

    unpadded_len=(len(librosa_output)//chunk_size)-forward_memory_mask_view

    return librosa_output,unpadded_len


    