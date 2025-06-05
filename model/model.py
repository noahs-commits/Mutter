from dataclasses import asdict
import torch
import torch.nn as nn
from model.hyperparameters import hyperparameters
from model.layers.AudioPreprocessor import AudioPreprocessor
from model.layers.LocalTransformer import LocalEncoder
from model.layers.PositionalEncoding import PositionalEncoding
import timed_tokenizer  

class transcription_model(nn.Module):
    def __init__(self, hp):
        super(transcription_model, self).__init__()

        self.hp=hp


        self.audio_preprocessor=AudioPreprocessor(hp.transformer_dim)
        self.audio_positional_encoding=PositionalEncoding(
            hp.transformer_dim,
            max_len=hp.max_audio_len,
            dropout=hp.dropout
        
        )
        self.local_audio_transformer=LocalEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hp.transformer_dim,
                nhead=hp.head_count,
                dim_feedforward=hp.dim_feedforward,
                activation=hp.activation,
                dropout=hp.dropout,
            ),
            num_layers=hp.layer_count,
            mask_size=hp.max_audio_lookback,
            max_len=hp.max_audio_len
        )

        self.embedding=nn.Embedding(
            hp.n_tokens,
            hp.transformer_dim,
        )

        self.text_positional_encoding=PositionalEncoding(
            hp.transformer_dim,
            max_len=hp.max_token,
            dropout=hp.dropout
        )

        self.word_decoder=nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hp.transformer_dim,
                nhead=hp.head_count,
                dim_feedforward=hp.dim_feedforward,
                activation=hp.activation,
                dropout=hp.dropout
            ),
            num_layers=hp.layer_count,
            norm=None
        )

        self.word_encoder=nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hp.transformer_dim,
                nhead=hp.head_count,
                dim_feedforward=hp.dim_feedforward,
                activation=hp.activation,
                dropout=hp.dropout
            ),
            num_layers=hp.layer_count,
            norm=None
        )

        self.final_layer=nn.Linear(hp.transformer_dim,hp.n_tokens)
        self.softmax=nn.Softmax(dim=-1)

        self.final_layer.weight = self.embedding.weight

    def forward(self, audio, text,tgt_mask_tensor,audio_memory_mask,can_be_end):
        audio=self.audio_preprocessor(audio)
        audio=audio.transpose(1,2)

        audio=self.audio_positional_encoding(audio)
        audio=self.local_audio_transformer(audio)

        text=self.embedding(text)
        text=self.text_positional_encoding(text)

        audio=audio.transpose(0,1)
        text=text.transpose(0,1)

        text=self.word_decoder.forward(text,audio, tgt_mask =tgt_mask_tensor,memory_mask=audio_memory_mask)
        
        seq_len = text.size(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=text.device), diagonal=1).bool()
        text = self.word_encoder(text, mask=causal_mask)
        

        text=self.final_layer(text)

        can_be_end=can_be_end.transpose(0,1)

        #prevent the model predicting the end token when there is more audio available
        text[...,timed_tokenizer.end_id][~can_be_end]=float("-inf")

        #prevent the model from requesting more audio when there is no more audio available
        text[...,timed_tokenizer.add_to_buffer_id][can_be_end]=float("-inf")

        return text
    def save(self, path,loss):

        torch.save({
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'hyperparameters': asdict(self.hp)#save in a dict format becuase dataclass needs be loaded with weights only set to False
        }, path)
    def load(path):
        # Load the checkpoint
        checkpoint = torch.load(path,weights_only=False)
        
        #save in a dict format becuase dataclass needs be loaded with weights only set to False
        hp_dict=checkpoint['hyperparameters']
        hp = hyperparameters(**hp_dict)

        state_dict = checkpoint['model_state_dict']


        # Initialize the model with the loaded hyperparameters
        model=transcription_model(hp)

        # Load the state dictionary into the model
        model.load_state_dict(state_dict)

        # Load the loss value
        loss=checkpoint['loss']
        return model,loss