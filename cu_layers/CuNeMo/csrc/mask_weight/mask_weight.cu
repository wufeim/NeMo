#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda.h>
#include <sstream>
#include <tuple>


__global__ void GatherIdxKernel(
    const int* input_valid, // (N)
    const int* sample_shifts, // (N, M) -> max_number of samples in each image
    const int* sample_indexs, // (N, M)
    const int* sample_n_verts, // (N, M)
    const int* cum_sum_n_verts_in_bank, // (Bank_count)
    const int N,
    const int M,
    const int K,
    int * object_idx, // (N, K_padded)
    int * source_vert_idx, // (N, K_padded)
    int * verts_start, // (N, K_padded)
    int * num_verts  // (N, K_padded)
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
        int curr_n_verts = 0;
        int next_shift = 0;
        for (int i = 0; i < M; i += 1) {
            next_shift = sample_shifts[n * M + i];
            if (next_shift == -1 || next_shift > k){
                // curr shift <= k < next shift
                // if k >= next_shift then curr_shift = next_shift
                break;
            }
            mesh_idx = sample_indexs[n * M + i];
            curr_n_verts = sample_n_verts[n * M + i];
            curr_shift = next_shift;
        }
        object_idx[pid] = mesh_idx;
        source_vert_idx[pid] = k - curr_shift;
        verts_start[pid] = cum_sum_n_verts_in_bank[mesh_idx];
        num_verts[pid] = curr_n_verts;
    }
}


__global__ void MaskWeightKernel(
    const int* object_idx, // (S, )
    const int* source_vert_idx, // (S, )
    const int* verts_start,  // (S, )
    const int* verts_end,  // (S, )
    const float* verts_dist_weight,  // (N_obj, V_padded, V_padded)
    const float weight_x, // weight -> cross class val
    const float weight_c, // weight -> clutter val
    const int clutter_start,
    const int S,
    const int K,
    const int V,
    float * out_weight // (S, K)
){
    const int num_threads = gridDim.x * blockDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int pid = tid; pid < S * K; pid += num_threads) {
        const int s = pid / K;
        const int k = pid % K;

        if (k >= verts_start[s] && k < verts_end[s]){
            const int obj_idx = object_idx[s];
            const int source_vert = source_vert_idx[s];
            const int target_vert = k - verts_start[s];
            out_weight[pid] = verts_dist_weight[obj_idx * V * V + source_vert * V + target_vert];
        }
        else if (k >= clutter_start) {
            out_weight[pid] = weight_c;
        }
        else{
            out_weight[pid] = weight_x;
        }
    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> GatherIdx(
    const at::Tensor& input_valid, // (N)
    const at::Tensor& sample_shifts, // (N, M) -> max_number of samples in each image
    const at::Tensor& sample_indexs, // (N, M)
    const at::Tensor& sample_n_verts, // (N, M)
    const at::Tensor& cum_sum_n_verts_in_bank, // (Bank_count)
    const int K
){
    at::cuda::CUDAGuard device_guard(input_valid.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = sample_shifts.size(0);
    const int M = sample_shifts.size(1);
    
    auto int_opts = input_valid.options().dtype(at::kInt);

    at::Tensor object_idx = at::zeros({N, K}, int_opts);
    at::Tensor source_vert_idx = at::zeros({N, K}, int_opts);
    at::Tensor verts_start = at::zeros({N, K}, int_opts);
    at::Tensor num_verts = at::zeros({N, K}, int_opts);

    const size_t blocks = 1024;
    const size_t threads = 64;

    GatherIdxKernel<<<blocks, threads, 0, stream>>>(
    input_valid.contiguous().data_ptr<int>(), // (N)
    sample_shifts.contiguous().data_ptr<int>(), // (N, M) -> max_number of samples in each image
    sample_indexs.contiguous().data_ptr<int>(), // (N, M)
    sample_n_verts.contiguous().data_ptr<int>(), // (N, M)
    cum_sum_n_verts_in_bank.contiguous().data_ptr<int>(), 
    N,
    M,
    K,
    object_idx.data_ptr<int>(), // (N, K_padded)
    source_vert_idx.data_ptr<int>(), // (N, K_padded)
    verts_start.data_ptr<int>(), // (N, K_padded)
    num_verts.data_ptr<int>()  // (N, K_padded)
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(object_idx, source_vert_idx, verts_start, num_verts);
}



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
){
    at::cuda::CUDAGuard device_guard(object_idx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int S = object_idx.size(0);
    const int V = verts_dist_weight.size(1);
    const int K = total_v;

    auto float_opts = verts_dist_weight.options().dtype(at::kFloat);

    at::Tensor out_weight = at::zeros({S, K}, float_opts);
    
    const size_t blocks = 1024;
    const size_t threads = 64;

    MaskWeightKernel<<<blocks, threads, 0, stream>>>(
        object_idx.contiguous().data_ptr<int>(), // (S, )
        source_vert_idx.contiguous().data_ptr<int>(), // (S, )
        verts_start.contiguous().data_ptr<int>(), // (S, )
        verts_end.contiguous().data_ptr<int>(), // (S, )
        verts_dist_weight.contiguous().data_ptr<float>(), // (N_obj, V_padded, V_padded)
        weight_x,
        weight_c,
        clutter_start,
        S,
        K,
        V,
        out_weight.data_ptr<float>()
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return out_weight;
}

