import torch
from CuNeMo import _C


def get_mask(verts_dist_weight, sample_indexs, mesh_n_list, total_size, K_padded, weight_x, weight_c, mask_sel=None, n_noise=0):
    basic_device = verts_dist_weight.device
    if not torch.is_tensor(mesh_n_list):
        mesh_n_list = torch.Tensor(mesh_n_list).type(torch.int32).to(basic_device)
    assert verts_dist_weight.device == sample_indexs.device == mesh_n_list.device
    assert verts_dist_weight.is_cuda
    assert verts_dist_weight.dim() == 3 
    verts_num, sample_shifts = get_vert_shifts(indexs=sample_indexs, mesh_n_list=mesh_n_list)

    sample_shifts = sample_shifts.type(torch.int32)
    sample_indexs = sample_indexs.type(torch.int32)
    verts_num = verts_num.type(torch.int32)

    input_valid = sample_shifts[:, -1] + verts_num[:, -1]
    cum_sum_n_verts_in_bank = (torch.cumsum(mesh_n_list, dim=0) - mesh_n_list).type(torch.int32)

    object_idx, source_vert_idx, verts_start, num_verts = _GatherIdx.apply(input_valid, sample_shifts, sample_indexs, verts_num, cum_sum_n_verts_in_bank, K_padded)

    if mask_sel is None:
        object_idx = object_idx.view(-1)
        source_vert_idx = source_vert_idx.view(-1)
        verts_start = verts_start.view(-1)
        num_verts = num_verts.view(-1)
    else:
        object_idx = object_idx[mask_sel]
        source_vert_idx = source_vert_idx[mask_sel]
        verts_start = verts_start[mask_sel]
        num_verts = num_verts[mask_sel]

    out_weight = _MaskWeight.apply(object_idx, source_vert_idx, verts_start, verts_start + num_verts, verts_dist_weight, weight_x, weight_c, total_size - n_noise, total_size)
    vert_index = torch.gather(torch.cumsum(mesh_n_list, dim=0) - mesh_n_list, dim=0, index=object_idx.long()) + source_vert_idx

    return out_weight, vert_index.long()


def get_vert_shifts(indexs, mesh_n_list):
    if not torch.is_tensor(mesh_n_list):
        mesh_n_list = torch.Tensor(mesh_n_list).type(indexs.dtype).to(indexs.device)
    valid_mask = torch.logical_not(indexs < 0).type(indexs.dtype)
    verts_num = valid_mask * torch.gather(mesh_n_list[None].expand(indexs.shape[0], -1), dim=1, index=indexs.long().clamp(min=0))
    sample_shifts = torch.cumsum(verts_num, dim=1) - verts_num
    return verts_num, sample_shifts


class _MaskWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
            object_idx,
            source_vert_idx, 
            verts_start,  
            verts_end,  
            verts_dist_weight,  
            weight_x, 
            weight_c, 
            clutter_start,
            total_v
        ):
        args = (
            object_idx,
            source_vert_idx, 
            verts_start,  
            verts_end,  
            verts_dist_weight,  
            weight_x, 
            weight_c, 
            clutter_start,
            total_v
        )
        out_weight = _C.mask_weight(*args)
        ctx.mark_non_differentiable(out_weight)
        return out_weight

# object_idx (N, K_padded) -> source object idx
# source_vert_idx (N, K_padded) -> verts id in that mesh
# verts_start (N, K_padded) -> verts start idx in K_padded
# num_verts (N, K_padded) -> number of verts of that mesh
class _GatherIdx(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
            input_valid, 
            sample_shifts, 
            sample_indexs, 
            sample_n_verts, 
            cum_sum_n_verts_in_bank, 
            K
        ):
        args = (
            input_valid, 
            sample_shifts, 
            sample_indexs, 
            sample_n_verts, 
            cum_sum_n_verts_in_bank,
            K
        )
        object_idx, source_vert_idx, verts_start, num_verts = _C.gather_idx(*args)
        ctx.mark_non_differentiable(object_idx, source_vert_idx, verts_start, num_verts)
        return object_idx, source_vert_idx, verts_start, num_verts


if __name__ == '__main__':
    device = 'cuda'

    kappas = {'pos':0, 'near':3, 'clutter': 4, 'class': 2}
    creater = MaskCreater(dist_thr=0.1, kappas=kappas, n_noise=2, verts_ori=[torch.rand((5, 3)), torch.rand((6, 3))], device='cuda')
    print(creater(sample_indexs=torch.Tensor([[1, 1], [0, 1]]).cuda().type(torch.int32), dtype_template=torch.ones(2, 12, 6)))

    exit(0)
    K = 12
    mesh_n_list = torch.Tensor([7, 5]).type(torch.int32).to(device)
    sample_indexs = torch.Tensor([[1, 0], [1, 1], [0, 1]],).type(torch.int32).to(device)

    verts_num, sample_shifts = get_vert_shifts(indexs=sample_indexs, mesh_n_list=mesh_n_list)
    input_valid = sample_shifts[:, -1] + verts_num[:, -1]

    object_idx, source_vert_idx, verts_start, num_verts = _GatherIdx.apply(input_valid.type(torch.int32), sample_shifts.type(torch.int32), sample_indexs.type(torch.int32), verts_num.type(torch.int32), 12)
    # print(torch.gather())
    print(torch.gather(torch.cumsum(mesh_n_list, dim=0) - mesh_n_list, dim=0, index=object_idx.view(-1).long()).view(*source_vert_idx.shape) + source_vert_idx)
    print(source_vert_idx)
    print(verts_start)
