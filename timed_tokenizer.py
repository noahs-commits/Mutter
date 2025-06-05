import tiktoken
gpt_tokenizer=tiktoken.get_encoding("r50k_base")


gpt_max_token=gpt_tokenizer.max_token_value

clear_buffer_id =gpt_max_token
add_to_buffer_id=gpt_max_token+1
end_id          =gpt_max_token+2
start_id        =gpt_max_token+3
ignore_id       =gpt_max_token+4

token_count=gpt_max_token+5


audio_tokenizer=tiktoken.Encoding(
    name="gpt2_audio_tokeniser",
    pat_str=gpt_tokenizer._pat_str,
    mergeable_ranks=gpt_tokenizer._mergeable_ranks,
    special_tokens={
        "<|clear_buffer|>": clear_buffer_id,
        "<|add_to_buffer|>": add_to_buffer_id,
        "<|end|>": end_id,
        "<|start|>": start_id,
        "<|ignore|>": ignore_id,
    }
)


