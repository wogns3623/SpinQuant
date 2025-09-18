import torch
from typing import Tuple, Optional


def adjust_rotation(R: torch.Tensor, inv=None, perm=None):
    if inv is not None:
        R = R @ inv.diag().to(R)

    if perm is not None:
        perm_mat = torch.eye(R.shape[-1]).to(R)[:, perm]
        R = R @ perm_mat

    return R


def find_adjust_matrices(
    x_rot: torch.Tensor,  # [batch_size, seqlen, hidden_dim]
    R: torch.Tensor,
    use_inv: bool,
    use_perm: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not use_inv and not use_perm:
        return None, None

    invert_list = None
    perm_order = None

    x_orig = x_rot.to(torch.float64) @ R.T.to(torch.float64)
    ma_channel_idx = x_orig.abs().max(dim=-2).values.argmax()
    ma_token_idx = x_orig.abs().max(dim=-1).values.argmax()
    x_for_sort = x_rot

    if use_inv:
        invert_list = R[ma_channel_idx].sign()
        x_for_sort = x_for_sort @ invert_list.diag().to(x_for_sort)
        x_for_sort = x_for_sort.abs()

    if use_perm:
        perm_order = x_for_sort[:, ma_token_idx].sort(descending=True).indices.squeeze()

    return invert_list, perm_order
