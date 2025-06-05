
from model import hyperparameters
from model.model import transcription_model


hp=hyperparameters.hyperparameters(
    transformer_dim=384,
    layer_count=4, 
    head_count=6,
    dim_feedforward=768,
    activation='gelu',
    dropout=0.1,
    max_audio_len=512,
    max_token=512,
    n_tokens=50000
)

model=transcription_model(hp)
total=0
params=[]
for name, param in model.named_parameters():
    if param.requires_grad:
        #print(f"{name}: {param.numel():,}")
        total+=param.numel()
        params.append((param.numel(), name))
print(f"Total parameters: {total:,}")
params.sort(reverse=True, key=lambda x: x[0])


for param,name in params:
    
    print(f"{name}: {param:,}")

print("======================================")


for name,child in model.named_children():
    print(f"{name}\t{sum(p.numel() for p in child.parameters() if p.requires_grad):,}")
