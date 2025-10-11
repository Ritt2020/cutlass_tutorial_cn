//
// 基本 Copy 示例
// 演示：
// 1. 主机端拷贝
// 2. 设备端拷贝（全局内存 → 共享内存 → 寄存器）
// 3. 多线程并行拷贝
//

#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

using namespace cute;

//
// 主机端基本拷贝示例
//
void test_host_copy() {
    std::cout << "\n=== 测试 1: 主机端基本拷贝 ===\n";
    
    // 源数据和目标数据
    float src_data[128];
    float dst_data[128];
    
    // 初始化源数据
    for (int i = 0; i < 128; ++i) {
        src_data[i] = i * 1.0f;
    }
    
    // 创建 Tensor
    auto src = make_tensor(&src_data[0], make_shape(Int<128>{}));
    auto dst = make_tensor(&dst_data[0], make_shape(Int<128>{}));
    
    // 拷贝
    copy(src, dst);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < 128; ++i) {
        if (dst_data[i] != src_data[i]) {
            success = false;
            std::cout << "错误: dst[" << i << "] = " << dst_data[i] 
                      << ", expected " << src_data[i] << "\n";
            break;
        }
    }
    
    if (success) {
        std::cout << "✓ 主机端拷贝成功！\n";
    }
}

//
// 设备端基本拷贝：全局内存 → 共享内存
//
__global__ void kernel_gmem_to_smem(float* gmem, float* result) {
    // 全局内存 Tensor
    auto g_tensor = make_tensor(make_gmem_ptr(gmem), make_shape(Int<256>{}));
    
    // 共享内存 Tensor
    __shared__ float smem[256];
    auto s_tensor = make_tensor(make_smem_ptr(smem), make_shape(Int<256>{}));
    
    // 256 个线程并行拷贝
    auto thr_layout = make_layout(Int<256>{});
    auto thr_g = local_partition(g_tensor, thr_layout, threadIdx.x);
    auto thr_s = local_partition(s_tensor, thr_layout, threadIdx.x);
    
    // 拷贝（每个线程拷贝 1 个元素）
    copy(thr_g, thr_s);
    
    // 同步，确保所有线程完成拷贝
    __syncthreads();
    
    // 将结果写回全局内存
    if (threadIdx.x < 256) {
        result[threadIdx.x] = smem[threadIdx.x];
    }
}

void test_device_copy() {
    std::cout << "\n=== 测试 2: 设备端拷贝 (全局内存 → 共享内存) ===\n";
    
    const int N = 256;
    
    // 主机端数据
    float* h_input = new float[N];
    float* h_output = new float[N];
    
    // 初始化输入
    for (int i = 0; i < N; ++i) {
        h_input[i] = i * 2.0f;
    }
    
    // 设备端数据
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动 kernel
    kernel_gmem_to_smem<<<1, 256>>>(d_input, d_output);
    cudaDeviceSynchronize();
    
    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_output[i] != h_input[i]) {
            success = false;
            std::cout << "错误: output[" << i << "] = " << h_output[i] 
                      << ", expected " << h_input[i] << "\n";
            break;
        }
    }
    
    if (success) {
        std::cout << "✓ 设备端拷贝成功！\n";
    }
    
    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

//
// 共享内存 → 寄存器 → 全局内存
//
__global__ void kernel_smem_to_rmem(float* gmem_in, float* gmem_out) {
    __shared__ float smem[128];
    
    // 第一步：全局内存 → 共享内存
    auto g_in = make_tensor(make_gmem_ptr(gmem_in), make_shape(Int<128>{}));
    auto s_tensor = make_tensor(make_smem_ptr(smem), make_shape(Int<128>{}));
    
    auto thr_layout = make_layout(Int<128>{});
    auto thr_g_in = local_partition(g_in, thr_layout, threadIdx.x);
    auto thr_s = local_partition(s_tensor, thr_layout, threadIdx.x);
    
    copy(thr_g_in, thr_s);
    __syncthreads();
    
    // 第二步：共享内存 → 寄存器
    // 使用 make_tensor_like 创建寄存器 Tensor
    auto r_tensor = make_tensor_like(thr_s);
    copy(thr_s, r_tensor);
    
    // 第三步：寄存器 → 全局内存（进行一些处理，如乘以 2）
    for (int i = 0; i < size(r_tensor); ++i) {
        r_tensor(i) = r_tensor(i) * 2.0f;
    }
    
    auto g_out = make_tensor(make_gmem_ptr(gmem_out), make_shape(Int<128>{}));
    auto thr_g_out = local_partition(g_out, thr_layout, threadIdx.x);
    
    copy(r_tensor, thr_g_out);
}

void test_smem_to_rmem() {
    std::cout << "\n=== 测试 3: 共享内存 → 寄存器 → 全局内存 ===\n";
    
    const int N = 128;
    
    // 主机端数据
    float* h_input = new float[N];
    float* h_output = new float[N];
    
    // 初始化输入
    for (int i = 0; i < N; ++i) {
        h_input[i] = i * 1.0f;
    }
    
    // 设备端数据
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动 kernel
    kernel_smem_to_rmem<<<1, 128>>>(d_input, d_output);
    cudaDeviceSynchronize();
    
    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果（应该是输入的 2 倍）
    bool success = true;
    for (int i = 0; i < N; ++i) {
        float expected = h_input[i] * 2.0f;
        if (h_output[i] != expected) {
            success = false;
            std::cout << "错误: output[" << i << "] = " << h_output[i] 
                      << ", expected " << expected << "\n";
            break;
        }
    }
    
    if (success) {
        std::cout << "✓ 共享内存 → 寄存器拷贝成功！\n";
    }
    
    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void kernel_2d_copy(float* gmem_in, float* gmem_out) {
    // M 行, N 列（注意：host 填充是 column-major: index = i + j * M）
    const int M = 8;
    const int N = 16;
    const int SIZE = M * N;

    // 创建 2D 张量（shape 使用 make_coord 表示二维）
    // 假设默认 layout 与 host 的 column-major 一致；如需显式 layout，请根据 Cute 版本传入 column_major layout。
    auto g_in = make_tensor(make_gmem_ptr(gmem_in), make_shape(make_coord(Int<M>{}, Int<N>{})));
    auto g_out = make_tensor(make_gmem_ptr(gmem_out), make_shape(make_coord(Int<M>{}, Int<N>{})));

    // 线程以线性方式分区整个 2D 区域（flat layout）
    // 每个线程处理 1 个 (或更多) 线性位置：thread_linear = threadIdx.x (+ blockDim.x*blockIdx.x 若需要)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;

    // 如果线程数 >= SIZE，这里把每个线程只做 1 个元素拷贝（更通用的方式是 while loop）
    // 我们用 local_partition 将 2D tensor 映射到一个 1D partition，保证 copy 的形状一致
    auto flat_layout = make_layout(Int<SIZE>{}); // flat linear layout for partitioning

    // 逐线程处理：把 tid 映射为该线程的元素位置（越界检查）
    for (int linear = tid; linear < SIZE; linear += num_threads) {
        // local_partition 会为该线程返回它负责的单元素子张量（1D）
        auto src_part = local_partition(g_in, flat_layout, linear);
        auto dst_part = local_partition(g_out, flat_layout, linear);

        // copy 单元素（CuTe 会在内部根据张量 layout 正确映射到 2D 内存地址）
        copy(src_part, dst_part);
    }
}

void test_2d_copy() {
    std::cout << "\n=== 测试 4: 二维 Tensor 拷贝 ===\n";

    const int M = 8;
    const int N = 16;
    const int SIZE = M * N;

    // 主机端数据
    float* h_input = new float[SIZE];
    float* h_output = new float[SIZE];

    // 初始化输入（列主序）
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            h_input[i + j * M] = i * 10 + j;
        }
    }

    // 设备端数据
    float *d_input, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));

    // 初始化输出为 0
    cudaMemset(d_output, 0, SIZE * sizeof(float));
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    // 用 1 个 block 并 128 线程也可，但更通用做法是：让 grid/ block 足够覆盖 SIZE
    const int threads = 128;
    const int blocks = (SIZE + threads - 1) / threads;
    kernel_2d_copy<<<blocks, threads>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool success = true;
    for (int i = 0; i < SIZE; ++i) {
        if (h_output[i] != h_input[i]) {
            success = false;
            std::cout << "错误: output[" << i << "] = " << h_output[i]
                      << ", expected " << h_input[i] << "\n";
            break;
        }
    }

    if (success) {
        std::cout << "✓ 二维 Tensor 拷贝成功！\n";
    }

    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    std::cout << "CuTe 基本 Copy 示例\n";
    std::cout << "====================\n";
    
    // 测试 1: 主机端拷贝
    test_host_copy();
    
    // 测试 2: 设备端拷贝
    test_device_copy();
    
    // 测试 3: 共享内存 → 寄存器
    test_smem_to_rmem();
    
    // 测试 4: 二维 Tensor 拷贝
    test_2d_copy();
    
    std::cout << "\n所有测试完成！\n";
    
    return 0;
}

