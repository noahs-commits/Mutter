import torch

def vectorized_slice_unfold(arr: torch.Tensor, indexes: list[int] | torch.Tensor, length: int) -> torch.Tensor:
    """
    Extracts multiple slices using torch.unfold.

    Args:
        arr (torch.Tensor): The input 2D tensor (e.g., shape [Rows, Columns]).
        indexes (list[int] | torch.Tensor): A list or 1D tensor of starting column indices.
        length (int): The desired length of each slice.

    Returns:
        torch.Tensor: A 3D tensor of shape [num_slices, Rows, length].
    """
    # --- Input Validation (Simplified for brevity, reuse checks from above) ---
    if arr.ndim != 2: raise ValueError("Input 'arr' must be 2D.")
    if length <= 0: raise ValueError("Length must be positive.")
    if isinstance(indexes, list):
        if not indexes: return torch.empty((0, arr.shape[0], length), dtype=arr.dtype, device=arr.device)
        indexes_tensor = torch.tensor(indexes, dtype=torch.long, device=arr.device)
    elif isinstance(indexes, torch.Tensor):
        if indexes.ndim != 1: raise ValueError("Indexes tensor must be 1D.")
        if indexes.numel() == 0: return torch.empty((0, arr.shape[0], length), dtype=arr.dtype, device=arr.device)
        indexes_tensor = indexes.to(dtype=torch.long, device=arr.device)
    else:
        raise TypeError("Indexes must be list or Tensor.")

    num_rows, num_cols = arr.shape
    max_idx = torch.max(indexes_tensor) if indexes_tensor.numel() > 0 else -1
    if length > num_cols or max_idx + length > num_cols or (indexes_tensor.numel() > 0 and torch.min(indexes_tensor) < 0):
         raise ValueError("Invalid indexes or length for tensor dimensions.")
    # --- Vectorized Slicing ---

    # 1. Create all possible overlapping slices (views) along columns
    # dimension=1 (columns), size=length (window size), step=1 (move one col at a time)
    # Shape: [Rows, num_possible_slices, length]
    # where num_possible_slices = num_cols - length + 1
    unfolded_arr = arr.unfold(dimension=1, size=length, step=1)

    # unfolded_arr[r, i, :] corresponds to arr[r, i : i + length]

    # 2. Select the slices corresponding to the desired start 'indexes'
    # We need slices starting at index `i` from the original `arr`, which corresponds
    # to the `i`-th slice in the `unfolded_arr`'s second dimension.
    # Shape: [Rows, num_slices, length]
    selected_unfolded = unfolded_arr[:, indexes_tensor, :] # Index along the 'num_possible_slices' dimension

    # 3. Permute dimensions to get the desired output shape [num_slices, Rows, length]
    # Current: (Rows, Slices, Length) -> Target: (Slices, Rows, Length)
    result = selected_unfolded.permute(1, 0, 2)

    # Ensure contiguous memory layout
    return result.contiguous()

# --- Example Usage (using the same 'arr' from Method 1) ---
rows, cols = 5, 30
arr = torch.arange(rows * cols).reshape(rows, cols)

print("\n" + "="*30 + "\nUsing Unfold Method\n" + "="*30)
indexes_list = [2, 10, 25]
slice_len = 4

result_unfold = vectorized_slice_unfold(arr, indexes_list, slice_len)

print(f"\nSlices for indexes={indexes_list}, length={slice_len}:")
print(f"Result shape: {result_unfold.shape}") # Should be [3, 5, 4]
print(result_unfold)

# Verify results are the same
#print(f"\nResults match? {torch.equal(result_tensor, result_unfold)}")