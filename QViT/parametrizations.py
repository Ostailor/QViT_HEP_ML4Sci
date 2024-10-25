import torch

# Converts an input array into quantum parameters, supports batch transformations.
def convert_array(X):
    """
    Converts the input array X into parameter angles for quantum operations.
    This function normalizes X and computes angles `alphas` which are used as
    parameters in quantum circuits.

    Args:
        X (Tensor): Input tensor to convert, with shape (..., N).

    Returns:
        Tensor: Tensor of computed angles `alphas` with shape (..., N-1).
    """
    # Initialize alphas tensor to store computed angles for each element in X
    alphas = torch.zeros(*X.shape[:-1], X.shape[-1] - 1)
    
    # Normalize X along the last dimension
    X_normd = X.clone() / (X**2).sum(axis=-1)[..., None].sqrt()
    
    # Compute angles for each element based on the normalized X
    for i in range(X.shape[-1] - 1):
        if i == 0:
            # Calculate the first angle based on the normalized first element
            alphas[..., i] = torch.acos(X_normd[..., i])
        elif i < (X.shape[-1] - 2):
            # Calculate intermediate angles using previous sine products
            alphas[..., i] = torch.acos(X_normd[..., i] / torch.prod(torch.sin(alphas[..., :i]), dim=-1))
        else:
            # Calculate the last angle using arctangent for the last two elements
            alphas[..., i] = torch.atan2(input=X_normd[..., -1], other=X_normd[..., -2])
    
    return alphas

# Converts a matrix into quantum parameters; does not support batch transformations.
def convert_matrix(X):
    """
    Converts a matrix X into parameter angles and magnitudes for quantum operations.
    
    Args:
        X (Tensor): Input matrix with shape (..., N, M).

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - mag_alphas: Angle parameters for row magnitudes.
            - alphas: Angle parameters for each element.
    """
    # Calculate magnitudes of each row and convert them to angle parameters
    mag_alphas = convert_array((X**2).sum(axis=1).sqrt())
    
    # Convert each element in X to angle parameters
    alphas = convert_array(X)
    
    return mag_alphas, alphas

# Splits colored data into patches and returns the patches as a flattened tensor.
def patcher_with_color(data, sh):
    """
    Splits multi-channel (e.g., color) data into patches and returns them as a flattened tensor.
    Designed for images with two color channels, but can be adjusted for other multi-channel data.

    Args:
        data (Tensor): Input data with shape (..., H, W, C).
        sh (Tuple[int, int]): Shape of each patch (rows, columns).

    Returns:
        Tensor: Tensor containing the flattened patches with shape (..., 2*rmax*cmax, r*c).
    """
    # Unpack the patch shape (rows, columns)
    r, c = sh
    
    # Determine the number of patches along each dimension
    rmax = data.shape[-3] // r
    cmax = data.shape[-2] // c
    
    # Initialize an empty tensor to hold the patches
    patched = torch.empty(*data.shape[:-3], 2 * rmax * cmax, r * c, device=data.device).type(torch.float32)
    n = 0  # Patch counter
    
    # Loop through each patch in the height, width, and channel dimensions
    for i in range(rmax):
        for j in range(cmax):
            for k in range(2):  # Assuming data has two color channels
                # Flatten and store each patch in the output tensor
                patched[..., n, :] = data[..., (i * r):(i * r + r), (j * c):(j * c + c), k].flatten(start_dim=-2)
                n += 1  # Increment patch counter
    
    return patched
