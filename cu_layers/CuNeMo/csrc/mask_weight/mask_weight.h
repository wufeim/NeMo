#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> GatherIdx(
    const at::Tensor& input_valid, // (N)
    const at::Tensor& sample_shifts, // (N, M) -> max_number of samples in each image
    const at::Tensor& sample_indexs, // (N, M)
    const at::Tensor& sample_n_verts, // (N, M)
    const at::Tensor& cum_sum_n_verts_in_bank, // (Bank_count)
    const int K
);

at::Tensor MaskWeight(
    const at::Tensor& object_idx, // (S, )
    const at::Tensor& source_vert_idx, // (S, )
    const at::Tensor& verts_start,  // (S, )
    const at::Tensor& verts_end,  // (S, )
    const at::Tensor& verts_dist_weight,  // (N_obj, V_padded, V_padded)
    const float weight_x, // weight -> cross class val
    const float weight_c, // weight -> clutter val
    const int clutter_start,
    const int total_v
);

#else
    AT_ERROR("Not compiled with GPU support");
#endif