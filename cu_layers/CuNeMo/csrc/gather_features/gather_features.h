#pragma once

#include <torch/extension.h>
#include <tuple>

#ifdef WITH_CUDA
std::tuple<at::Tensor, at::Tensor, at::Tensor> GatherFeatures(
    const at::Tensor& input_features, // (N, K_padded, C)
    const at::Tensor& input_weights, // (N, K_padded, )
    const at::Tensor& input_valid, // (N) -> number of meshes in bank
    const at::Tensor& bank_shifts, // (L) -> number of meshes in bank
    const at::Tensor& sample_shifts, // (N, M) -> max_number of samples in each image
    const at::Tensor& sample_indexs, // (N, M)
    const int KK // Max size of the feature bank
);

std::tuple<at::Tensor, at::Tensor> GatherFeaturesBackward(
    const at::Tensor& gard_gathered_features, // (KK, C)
    const at::Tensor& gard_gathered_weight, // (KK, )
    const at::Tensor& input_features, // (N, K_padded, C)
    const at::Tensor& input_weights, // (N, K_padded, )
    const at::Tensor& forward_mapping // (N, K_padded)
);

#else
    AT_ERROR("Not compiled with GPU support");
#endif