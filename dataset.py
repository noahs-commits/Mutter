from io import StringIO
import json
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader


from datasets import load_dataset
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import stable_whisper
import itertools
from dataclasses import dataclass
import pickle
from tqdm import tqdm
import sys, os

class NullIO(StringIO):
    def write(self, txt):
       pass

@dataclass
class WordTiming():
    start: int
    end: int
    word: str


    
class time_stamped_audio(Dataset):
    def __init__(self,split):
        super().__init__()

        self.base_data_set=load_dataset("mozilla-foundation/common_voice_11_0", "en", split=split)
        self.model=None
        self.data_base = sqlite3.connect("C:\\Users\\noahk\\OneDrive\\Desktop\\Python\\audio_data_set\\whisper_cache.db")
        self.data_base_cursor = self.data_base.cursor()
        self.data_base_cursor.execute("CREATE TABLE IF NOT EXISTS cache (input TEXT PRIMARY KEY, output BLOB)")
        self.data_base.commit()


    def __len__(self):
        return len(self.base_data_set)
    def get_whisper_output(self,audio_arr,path,text):
        self.data_base_cursor.execute("SELECT output FROM cache WHERE input=?", (path,))
        row = self.data_base_cursor.fetchone()
        if row:
            result = pickle.loads(row[0])
            return result
        
        #print("Computing...")
        if self.model is None:
            self.model = stable_whisper.load_model('base')
        try:
            #for some reason the libbars verbose feture is not working so I am redirecting the stdout to /dev/null to hide the output
            original_stdout = sys.stdout
            sys.stdout = NullIO()
            result = self.model.align(path,text,language='en',verbose=None)
        finally:
            sys.stdout = original_stdout
        
        self.data_base_cursor.execute("INSERT INTO cache (input, output) VALUES (?, ?)", (path, pickle.dumps(result)))
        self.data_base.commit()
        return result
    def __getitem__(self, index):
        
        
        test_case_dict=self.base_data_set[index]
        whisper_output=self.get_whisper_output(test_case_dict["audio"]["array"],test_case_dict["audio"]["path"],test_case_dict["sentence"])

        timed_words=[word for s in whisper_output.segments for word in s.words]


        return (timed_words,test_case_dict["sentence"],test_case_dict["audio"])
    
if __name__ == "__main__":
    dataset = time_stamped_audio("test")

    for _ in tqdm(dataset):
        pass