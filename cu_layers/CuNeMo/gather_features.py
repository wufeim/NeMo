import torch
from CuNeMo import _C


def gather_features(features, weights, sample_indexs, mesh_n_list):
    """
    Sample features from a set of concatnated feature list on each image. The input features is expected to be sample from a concatenated meshes that for each image:
    mesh_img_k = Meshes(verts=[vert_mesh_i0, verts_mesh_i1, ...., verts_mesh_im], ...)
    Output features is sorted as Meshes(verts=[vert_mesh_0, vert_mesh_1, ..., vert_mesh_l])
    features: Tensor - (N, K, C), torch.float32, input features to sample
    weights: Tensor - (N, K), torch.float32, weight of feature to sample
    sample_indexs: Tensor - (N, M), torch.int32, indicates the sort of mesh on each image -> [[i0, i1, ..., im], ...]
    mesh_n_list: List - [int]  or   Tensor - (L), torch.int32, the number of verts of mesh in the bank
    """
    basic_device = features.device
    if not torch.is_tensor(mesh_n_list):
        mesh_n_list = torch.Tensor(mesh_n_list).type(torch.int32).to(basic_device)
    assert basic_device == weights.device == sample_indexs.device == mesh_n_list.device
    assert features.is_cuda
    assert features.dim() == 3 and weights.dim() == 2 and sample_indexs.dim() == 2
    total_verts = mesh_n_list.sum()

    assert 0 <= sample_indexs.max() < mesh_n_list.shape[0]
    assert sample_indexs.dtype == mesh_n_list.dtype == torch.int32

    samples_cum_num = torch.cumsum(torch.gather(mesh_n_list[None].expand(sample_indexs.shape[0], -1), dim=1, index=sample_indexs.long()), dim=1).type(torch.int32)
    input_valid = samples_cum_num[:, -1]
    sample_shifts = torch.cat((torch.zeros((samples_cum_num.shape[0], 1), dtype=torch.int32, device=basic_device), samples_cum_num[:, :-1]), dim=1)

    bank_shifts = torch.cumsum(mesh_n_list, dim=0).type(torch.int32) - mesh_n_list

    gathered_features, gathered_weights, forward_mapping = _GatherFeatures.apply(features, weights, input_valid, bank_shifts, sample_shifts, sample_indexs, total_verts)
    return gathered_features, gathered_weights


class _GatherFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input_features, # (N, K_padded, C)
                input_weights, # (N, K_padded)
                input_valid, # (N, )
                bank_shifts, # (L, )
                sample_shifts, # (N, M)
                sample_indexs, # (N, M)
                KK
        ):
        args = (
                input_features,
                input_weights,
                input_valid,
                bank_shifts,
                sample_shifts,
                sample_indexs,
                KK
        )
        gathered_features, gathered_weights, forward_mapping = _C.gather_features(*args)
        ctx.save_for_backward(input_features, input_weights, forward_mapping)
        ctx.mark_non_differentiable(forward_mapping)
        return gathered_features, gathered_weights, forward_mapping

    @staticmethod
    def backward(ctx,
                 gard_gathered_features,
                 grad_gathered_weights,
                 grad_forward_mapping
        ):
        input_features, input_weights, forward_mapping = ctx.saved_tensors
        args = (
                gard_gathered_features,
                grad_gathered_weights,
                input_features,
                input_weights,
                forward_mapping,
        )
        gard_input_features, gard_input_weights = _C.gather_features_backward(*args)
        return gard_input_features, gard_input_weights, None, None, None, None, None

if __name__ == '__main__':
    device = 'cuda'
    input_features = torch.rand((1, 12, 3), device=device)
    input_weights = torch.ones((1, 12), device=device)
    input_valid = torch.Tensor([12]).type(torch.int32).to(device)
    bank_shifts = torch.Tensor([0, 7],).type(torch.int32).to(device)
    sample_shifts = torch.Tensor([[0, 5]],).type(torch.int32).to(device)
    sample_indexs = torch.Tensor([[1, 0]],).type(torch.int32).to(device)

    
#     input_features = torch.nn.Parameter(input_features)
#     input_weights = torch.nn.Parameter(input_weights)
    gathered_features, gathered_counts, forward_mapping = _GatherFeatures.apply(input_features, input_weights, input_valid, bank_shifts, sample_shifts, sample_indexs, 12)

#     gathered_features.sum().backward()
    mesh_n_list = [7, 5]
    gathered_features1, gathered_counts1 = gather_features(input_features, input_weights, sample_indexs, mesh_n_list)

