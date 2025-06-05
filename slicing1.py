import torch

def vectorized_slice_indexing(arr: torch.Tensor, indexes: list[int] | torch.Tensor, length: int) -> torch.Tensor:
    """
    Extracts multiple slices of a fixed length from a 2D tensor based on starting indices.

    Args:
        arr (torch.Tensor): The input 2D tensor (e.g., shape [Rows, Columns]).
        indexes (list[int] | torch.Tensor): A list or 1D tensor of starting column indices
                                            for each slice.
        length (int): The desired length of each slice along the column dimension.

    Returns:
        torch.Tensor: A 3D tensor where the first dimension corresponds to the slices.
                      Shape: [num_slices, Rows, length].
                      The i-th slice is equivalent to arr[:, indexes[i] : indexes[i] + length].

    Raises:
        ValueError: If inputs are invalid (e.g., arr not 2D, indexes out of bounds, length <= 0).
        TypeError: If inputs have incorrect types.
    """
    # --- Input Validation ---
    if not isinstance(arr, torch.Tensor):
        raise TypeError("Input 'arr' must be a PyTorch Tensor.")
    if arr.ndim != 2:
        raise ValueError(f"Input 'arr' must be a 2D tensor, but got {arr.ndim} dimensions.")
    if not isinstance(length, int) or length <= 0:
        raise ValueError(f"Input 'length' must be a positive integer, but got {length}.")
    if isinstance(indexes, list):
        if not indexes: # Handle empty list case
             return torch.empty((0, arr.shape[0], length), dtype=arr.dtype, device=arr.device)
        indexes_tensor = torch.tensor(indexes, dtype=torch.long, device=arr.device)
    elif isinstance(indexes, torch.Tensor):
        if indexes.ndim != 1:
             raise ValueError(f"Input 'indexes' tensor must be 1D, but got {indexes.ndim} dimensions.")
        if indexes.numel() == 0: # Handle empty tensor case
             return torch.empty((0, arr.shape[0], length), dtype=arr.dtype, device=arr.device)
        indexes_tensor = indexes.to(dtype=torch.long, device=arr.device) # Ensure correct type/device
    else:
        raise TypeError("Input 'indexes' must be a list of integers or a 1D PyTorch Tensor.")

    num_rows, num_cols = arr.shape
    num_slices = indexes_tensor.shape[0]

    # Check index bounds
    min_index = torch.min(indexes_tensor)
    max_required_index = torch.max(indexes_tensor) + length -1 # Last index needed

    if min_index < 0:
        raise ValueError(f"Minimum index {min_index.item()} is out of bounds (must be >= 0).")
    if max_required_index >= num_cols:
         raise ValueError(
             f"Maximum required index {max_required_index.item()} "
             f"(from index {torch.max(indexes_tensor).item()} + length {length} - 1) "
             f"is out of bounds for tensor with {num_cols} columns."
         )
    # --- Vectorized Slicing ---

    # 1. Create offsets for the columns within each slice
    # Shape: [length] -> [0, 1, ..., length-1]
    col_offsets = torch.arange(length, device=arr.device)

    # 2. Broadcast 'indexes' and 'col_offsets' to get all column indices
    # indexes_tensor shape: [num_slices] -> unsqueeze -> [num_slices, 1]
    # col_offsets shape:                       [length]
    # Broadcasting `+` results in shape: [num_slices, length]
    # Example: indexes=[10, 20], length=3
    # col_indices = [[10, 11, 12],
    #                [20, 21, 22]]
    col_indices = indexes_tensor.unsqueeze(1) + col_offsets

    # 3. Use advanced indexing to select columns
    # arr[:, col_indices]
    # arr shape: [Rows, Columns]
    # col_indices shape: [num_slices, length]
    # Result shape according to PyTorch indexing rules: [Rows, num_slices, length]
    print(f"{col_indices=}")
    sliced_data = arr[:, col_indices]

    # 4. Permute dimensions to get the desired output shape [num_slices, Rows, length]
    # Current: (Rows, Slices, Length) -> Target: (Slices, Rows, Length)
    # Indices:    0      1       2             ->      1      0       2
    result = sliced_data.permute(1, 0, 2)

    # Ensure contiguous memory layout for potentially better performance downstream
    return result.contiguous()

# --- Example Usage ---
rows, cols = 5, 30
arr = torch.arange(rows * cols).reshape(rows, cols)
print("Original Tensor (arr):")
print(arr)
print("-" * 20)

indexes_list = [2, 10, 25]
slice_len = 4

result_tensor = vectorized_slice_indexing(arr, indexes_list, slice_len)

print(f"\nSlices for indexes={indexes_list}, length={slice_len}:")
print(f"Result shape: {result_tensor.shape}") # Should be [3, 5, 4]
print(result_tensor)
print("-" * 20)

print("Verification (manual slicing):")
print("Slice 0 (arr[:, 2:6]):")
print(arr[:, 2:2+slice_len])
print("Slice 1 (arr[:, 10:14]):")
print(arr[:, 10:10+slice_len])
print("Slice 2 (arr[:, 25:29]):")
print(arr[:, 25:25+slice_len])

# Example with tensor input for indexes
indexes_tensor = torch.tensor([0, 5, 15], device=arr.device)
result_tensor_2 = vectorized_slice_indexing(arr, indexes_tensor, 3)
print(f"\nSlices for indexes={indexes_tensor.tolist()}, length=3:")
print(f"Result shape: {result_tensor_2.shape}") # Should be [3, 5, 3]
print(result_tensor_2)

# Example with edge case: empty indexes
result_empty = vectorized_slice_indexing(arr, [], 5)
print(f"\nResult for empty indexes: {result_empty.shape}") # Should be [0, 5, 5]