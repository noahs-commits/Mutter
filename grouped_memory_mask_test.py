import torch
import torch.nn as nn

# --- Configuration ---
batch_size = 4
d_model = 512
nhead = 8
num_decoder_layers = 1 # Using a single layer for simplicity
dim_feedforward = 2048
dropout = 0.1

tgt_len = 10 # Target sequence length
src_len = 15 # Source (memory) sequence length

# --- Dummy Input Data ---
# Target input for the decoder
tgt = torch.rand(batch_size, tgt_len, d_model)
# Memory input (output from the encoder)
memory = torch.rand(batch_size, src_len, d_model)

# --- Assume PyTorch's batch_first=False convention for Transformer layers ---
# Transformer layers expect (SeqLen, Batch, Dim) by default
tgt = tgt.transpose(0, 1) # Shape: (tgt_len, batch_size, d_model)
memory = memory.transpose(0, 1) # Shape: (src_len, batch_size, d_model)

# --- Create Per-Item Memory Masks ---
# Let's create N=batch_size different masks.
# Each mask should be (T, S) = (tgt_len, src_len)
# Example: Mask out different source positions for different batch items.
# For simplicity, let's create boolean masks (True means masked)
individual_masks = []
for i in range(batch_size):
    mask = torch.zeros(tgt_len, src_len, dtype=torch.bool)
    # Example masking logic: mask the first 'i+1' source positions for batch item 'i'
    # (This is just a placeholder, replace with your actual masking logic)
    if i < src_len:
         mask[:, :i+1] = True # Mask first i+1 columns (source positions)
    individual_masks.append(mask)

# Stack the individual masks along the batch dimension
# Shape: (batch_size, tgt_len, src_len)
batched_memory_mask = torch.stack(individual_masks, dim=0)
print(f"Shape of stacked individual masks: {batched_memory_mask.shape}") # Should be (N, T, S)

# --- Expand the mask for attention heads ---
# MultiheadAttention expects (N * num_heads, T, S)
# We can achieve this by repeating the mask for each head.

# 1. Unsqueeze to add a head dimension: (N, 1, T, S)
expanded_mask = batched_memory_mask.unsqueeze(1)

# 2. Repeat along the new head dimension: (N, num_heads, T, S)
expanded_mask = expanded_mask.repeat(1, nhead, 1, 1)

# 3. Reshape to combine batch and head dimensions: (N * num_heads, T, S)
final_memory_mask = expanded_mask.view(batch_size * nhead, tgt_len, src_len)
print(f"Shape of final memory_mask for MHA: {final_memory_mask.shape}") # Should be (N*nhead, T, S)

# --- Alternative using Float masks (-inf for masked) ---
# If using float masks (required by some PyTorch versions/settings)
# Where 0.0 means attend, -inf means mask.
# float_mask = torch.zeros_like(batched_memory_mask, dtype=torch.float32)
# float_mask[batched_memory_mask] = -float('inf') # Apply -inf where boolean mask is True
# final_memory_mask_float = float_mask.unsqueeze(1).repeat(1, nhead, 1, 1).view(batch_size * nhead, tgt_len, src_len)
# print(f"Shape of final float memory_mask for MHA: {final_memory_mask_float.shape}")

# --- Instantiate and Use Decoder Layer ---
decoder_layer = nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=False # Match input format
)
decoder_layer.eval() # Set to evaluation mode for stable output

# Pass the constructed 3D mask to the forward method
# Note: Depending on the PyTorch version and exact internal implementation,
# MultiheadAttention might accept (N, T, S) and handle broadcasting internally,
# but providing the explicit (N * num_heads, T, S) is the safest approach based on docs.
# Let's use the boolean mask directly, as MHA often handles converting it.
# If you encounter issues, try the float version.
with torch.no_grad():
    output = decoder_layer(
        tgt,                # Target input (T, N, E)
        memory,             # Memory input (S, N, E)
        tgt_mask=None,      # Optional: Standard causal mask for target self-attention
        memory_mask=final_memory_mask # Your per-item mask expanded for heads (N*nhead, T, S)
    )

print(f"Output shape from decoder layer: {output.shape}") # Should be (T, N, E)

# If using the full nn.Transformer or nn.TransformerDecoder, you would pass
# final_memory_mask (or final_memory_mask_float) similarly to their forward methods.