#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda.h>
#include <sstream>
#include <tuple>


__device__ void inline weightedvectoratom(
    const float* source,
    const float weight,
    float* target,
    int32_t size
){
    for (int i = 0; i < size; ++i){
        atomicAdd(target + i, source[i] * weight);
    }
}


__global__ void GatherFeaturesKernel(
    const float* input_features, // (N, K_padded, C)
    const float* input_weights, // (N, K_padded, )
    const int* input_valid, // (N)
    const int* bank_shifts, // (L) -> number of meshes in bank
    const int* sample_shifts, // (N, M) -> max_number of samples in each image
    const int* sample_indexs, // (N, M)
    const int N,
    const int M,
    const int K,
    const int C,
    float * gathered_features,
    float * gathered_weights,
    int * forward_mapping
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < N * K; pid += num_threads) {
        const int n = pid / K;
        const int k = pid % K;
        if (k >= input_valid[n]){
            continue;
        }

        int mesh_idx = 0;
        int curr_shift = 0;
        int next_shift = 0;
        for (int i = 0; i < M; i += 1) {
            next_shift = sample_shifts[n * M + i];
            if (next_shift == -1 || next_shift > k){
                // curr shift <= k < next shift
                // if k >= next_shift then curr_shift = next_shift
                break;
            }
            mesh_idx = sample_indexs[n * M + i];
            curr_shift = next_shift;
        }

        int bank_vert_idx = k - curr_shift + bank_shifts[mesh_idx];
        float vert_weight = input_weights[pid];
        forward_mapping[pid] = bank_vert_idx;
        atomicAdd(gathered_weights + bank_vert_idx, vert_weight);
        weightedvectoratom(input_features + pid * C, vert_weight, gathered_features + bank_vert_idx * C, C);
    }
}


__global__ void GatherFeaturesBackwardKernel(
    const float* gard_gathered_features, // (KK, C)
    const float* gard_gathered_weight, // (KK, )
    const float* input_features, // (N, K_padded, C)
    const float* input_weights, // (N, K_padded, )
    const int* forward_mapping, // (N, K_padded)
    const int N,
    const int K,
    const int C,
    float * gard_input_features, // (N, K_padded, C)
    float * gard_input_weights // (N, K_padded, )
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < N * K; pid += num_threads) {
        if (forward_mapping[pid] < 0){
            continue;
        }
        int bank_vert_idx = forward_mapping[pid];
        float to_add_weight = gard_gathered_weight[bank_vert_idx];
        for (int c = 0; c < C; c++){
            to_add_weight += gard_gathered_features[bank_vert_idx * C + c] * input_features[pid * C + c];
        }
        atomicAdd(gard_input_weights + pid, to_add_weight);
        weightedvectoratom(gard_gathered_features + bank_vert_idx * C, input_weights[pid], gard_input_features + pid * C, C);
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> GatherFeatures(
    const at::Tensor& input_features, // (N, K_padded, C)
    const at::Tensor& input_weights, // (N, K_padded, )
    const at::Tensor& input_valid, // (N) -> number of meshes in bank
    const at::Tensor& bank_shifts, // (L) -> number of meshes in bank
    const at::Tensor& sample_shifts, // (N, M) -> max_number of samples in each image
    const at::Tensor& sample_indexs, // (N, M)
    const int KK // Max size of the feature bank
){
    at::cuda::CUDAGuard device_guard(input_features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = input_features.size(0);
    const int K = input_features.size(1);
    const int C = input_features.size(2);
    const int M = sample_shifts.size(1);

    auto float_opts = input_features.options().dtype(at::kFloat);
    auto int_opts = input_features.options().dtype(at::kInt);

    at::Tensor gathered_features = at::zeros({KK, C}, float_opts);
    at::Tensor gathered_weights = at::zeros({KK}, float_opts);
    at::Tensor forward_mapping = at::full({N, K}, -1, int_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    GatherFeaturesKernel<<<blocks, threads, 0, stream>>>(
        input_features.contiguous().data_ptr<float>(), // (N, K_padded, C)
        input_weights.contiguous().data_ptr<float>(), // (N, K_padded, C)
        input_valid.contiguous().data_ptr<int>(), // (N)
        bank_shifts.contiguous().data_ptr<int>(), // (L) -> number of meshes in bank
        sample_shifts.contiguous().data_ptr<int>(), // (N, M) -> max_number of samples in each image
        sample_indexs.contiguous().data_ptr<int>(), // (N, M)
        N,
        M,
        K,
        C,
        gathered_features.data_ptr<float>(),
        gathered_weights.data_ptr<float>(),
        forward_mapping.data_ptr<int>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(gathered_features, gathered_weights, forward_mapping);
}


std::tuple<at::Tensor, at::Tensor> GatherFeaturesBackward(
    const at::Tensor& gard_gathered_features, // (KK, C)
    const at::Tensor& gard_gathered_weight, // (KK, )
    const at::Tensor& input_features, // (N, K_padded, C)
    const at::Tensor& input_weights, // (N, K_padded, )
    const at::Tensor& forward_mapping // (N, K_padded)
){
    at::cuda::CUDAGuard device_guard(input_features.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = input_features.size(0);
    const int K = input_features.size(1);
    const int C = input_features.size(2);

    auto float_opts = input_features.options().dtype(at::kFloat);

    at::Tensor gard_input_features = at::zeros({N, K, C}, float_opts);
    at::Tensor gard_input_weights = at::zeros({N, K}, float_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    GatherFeaturesBackwardKernel<<<blocks, threads, 0, stream>>>(
        gard_gathered_features.contiguous().data_ptr<float>(), // (KK, C)
        gard_gathered_weight.contiguous().data_ptr<float>(), // (KK, )
        input_features.contiguous().data_ptr<float>(), // (N, K_padded, C)
        input_weights.contiguous().data_ptr<float>(), // (N, K_padded, )
        forward_mapping.contiguous().data_ptr<int>(), // (N, K_padded)
        N,
        K,
        C,
        gard_input_features.data_ptr<float>(),
        gard_input_weights.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(gard_input_features, gard_input_weights);
}