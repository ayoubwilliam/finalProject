import torch
import torch.nn.functional as F
from constants import DEVICE


def get_structural_element_kernel(kernel_size):
    """ Creates a binary ball kernel for 3D morphology on GPU. """
    r = kernel_size // 2
    x = torch.arange(-r, r + 1, device=DEVICE)
    y = torch.arange(-r, r + 1, device=DEVICE)
    z = torch.arange(-r, r + 1, device=DEVICE)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    kernel = (xx ** 2 + yy ** 2 + zz ** 2) <= r ** 2
    return kernel.float().unsqueeze(0).unsqueeze(0)


def morph_operation(mask_roi, kernel, operation='dilation'):
    """ Performs binary dilation or erosion using 3D convolution. """
    inp = mask_roi.float().unsqueeze(0).unsqueeze(0)
    pad = kernel.shape[-1] // 2
    out = F.conv3d(inp, kernel, padding=pad)

    if operation == 'dilation':
        return (out > 0).squeeze(0).squeeze(0)
    elif operation == 'erosion':
        kernel_sum = kernel.sum()
        # Use epsilon for float comparison safety
        return (out >= (kernel_sum - 0.1)).squeeze(0).squeeze(0)


def get_expanded_roi_torch(mask, kernel_size, shape):
    """ Expands the bounding box by kernel_size and clips to volume shape. """
    nonzero = torch.nonzero(mask)
    if nonzero.numel() == 0:
        return 0, 0, 0, 0, 0, 0

    min_coords = nonzero.min(dim=0).values
    max_coords = nonzero.max(dim=0).values + 1

    x0, x1 = min_coords[0].item(), max_coords[0].item()
    y0, y1 = min_coords[1].item(), max_coords[1].item()
    z0, z1 = min_coords[2].item(), max_coords[2].item()

    x0 = max(0, x0 - kernel_size)
    y0 = max(0, y0 - kernel_size)
    z0 = max(0, z0 - kernel_size)
    x1 = min(shape[0], x1 + kernel_size)
    y1 = min(shape[1], y1 + kernel_size)
    z1 = min(shape[2], z1 + kernel_size)

    return x0, x1, y0, y1, z0, z1


def apply_pooling(data_tensor, mask_tensor, kernel_size):
    """
    Applies pooling to the boundary shell of the object on GPU.
    Returns: Updated Data Tensor, Full Dilated Mask Tensor
    """
    # 1. Get ROI
    x0, x1, y0, y1, z0, z1 = get_expanded_roi_torch(mask_tensor, kernel_size, data_tensor.shape)

    # 2. Extract Views (Slices)
    local_mask = mask_tensor[x0:x1, y0:y1, z0:z1]
    local_data = data_tensor[x0:x1, y0:y1, z0:z1]

    # 3. Create Shell Mask (Dilation XOR Erosion)
    morph_kernel = get_structural_element_kernel(kernel_size)
    dilated = morph_operation(local_mask, morph_kernel, 'dilation')
    eroded = morph_operation(local_mask, morph_kernel, 'erosion')
    boundary_shell_mask = dilated ^ eroded

    # 4. Pooling
    pool_input = local_data.unsqueeze(0).unsqueeze(0)
    # Using stride=2 as per your original request
    pooled_res = F.avg_pool3d(pool_input, kernel_size, stride=2)
    pooled_resized = F.interpolate(pooled_res, size=local_data.shape, mode="nearest")
    pooled_view = pooled_resized.squeeze(0).squeeze(0)

    # 5. Update Data (In-Place on the specific shell pixels)
    # We clone local_data to avoid modifying the input tensor implicitly if we want pure functional style,
    # but for memory efficiency in 'add_mass', modifying the view updates the main tensor.
    local_data[boundary_shell_mask] = pooled_view[boundary_shell_mask]

    # 6. Construct full dilated mask (for correct_mask_by_seg later)
    full_dilated_mask = torch.zeros_like(mask_tensor)
    full_dilated_mask[x0:x1, y0:y1, z0:z1] = dilated

    return data_tensor, full_dilated_mask