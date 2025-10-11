//
// Copy_if 条件拷贝示例
// 演示：
// 1. 基本的 copy_if 用法
// 2. 边界处理场景
// 3. 条件筛选数据
//

#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

using namespace cute;

//
// 主机端条件拷贝示例
//
void test_host_copy_if() {
    std::cout << "\n=== 测试 1: 主机端条件拷贝 ===\n";
    
    const int N = 128;
    
    // 源数据、目标数据、谓词
    float src[N];
    float dst[N];
    bool mask[N];
    
    // 初始化
    for (int i = 0; i < N; ++i) {
        src[i] = i * 1.0f;
        mask[i] = (i % 2 == 0);  // 只拷贝偶数索引
        dst[i] = -1.0f;  // 初始化为 -1
    }
    
    // 创建 Tensor
    auto src_tensor = make_tensor(&src[0], make_shape(Int<N>{}));
    auto dst_tensor = make_tensor(&dst[0], make_shape(Int<N>{}));
    auto mask_tensor = make_tensor(&mask[0], make_shape(Int<N>{}));
    
    // 条件拷贝（注意参数顺序：pred, src, dst）
    copy_if(mask_tensor, src_tensor, dst_tensor);
    
    // 验证：只有偶数索引被拷贝
    bool success = true;
    int copied_count = 0;
    int skipped_count = 0;
    
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            if (dst[i] == src[i]) {
                copied_count++;
            } else {
                success = false;
                std::cout << "错误: dst[" << i << "] = " << dst[i] 
                          << ", expected " << src[i] << " (应该被拷贝)\n";
            }
        } else {
            if (dst[i] == -1.0f) {
                skipped_count++;
            } else {
                success = false;
                std::cout << "错误: dst[" << i << "] = " << dst[i] 
                          << ", expected -1.0 (不应该被拷贝)\n";
            }
        }
    }
    
    if (success) {
        std::cout << "✓ 条件拷贝成功！\n";
        std::cout << "  拷贝了 " << copied_count << " 个元素\n";
        std::cout << "  跳过了 " << skipped_count << " 个元素\n";
    }
}

//
// 设备端条件拷贝：只拷贝正数
//
__global__ void kernel_copy_positive(float* src, float* dst, int* count) {
    const int N = 256;
    
    // 全局内存 Tensor
    auto g_src = make_tensor(make_gmem_ptr(src), make_shape(Int<N>{}));
    auto g_dst = make_tensor(make_gmem_ptr(dst), make_shape(Int<N>{}));
    
    // 共享内存用于谓词
    __shared__ bool pred[N];
    
    // 生成谓词：只拷贝正数
    if (threadIdx.x < N) {
        pred[threadIdx.x] = (src[threadIdx.x] > 0.0f);
    }
    __syncthreads();
    
    auto pred_tensor = make_tensor(&pred[0], make_shape(Int<N>{}));
    
    // 分区
    auto thr_layout = make_layout(Int<256>{});
    auto thr_src = local_partition(g_src, thr_layout, threadIdx.x);
    auto thr_dst = local_partition(g_dst, thr_layout, threadIdx.x);
    auto thr_pred = local_partition(pred_tensor, thr_layout, threadIdx.x);
    
    // 条件拷贝（注意参数顺序：pred, src, dst）
    copy_if(thr_pred, thr_src, thr_dst);
    
    // 统计拷贝的元素数量
    __syncthreads();
    if (threadIdx.x == 0) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            if (pred[i]) cnt++;
        }
        *count = cnt;
    }
}

void test_device_copy_positive() {
    std::cout << "\n=== 测试 2: 设备端条件拷贝（只拷贝正数）===\n";
    
    const int N = 256;
    
    // 主机端数据
    float* h_src = new float[N];
    float* h_dst = new float[N];
    
    // 初始化：一半正数，一半负数
    for (int i = 0; i < N; ++i) {
        h_src[i] = (i % 2 == 0) ? (float)i : -(float)i;
        h_dst[i] = -999.0f;  // 初始化为特殊值
    }
    
    // 设备端数据
    float *d_src, *d_dst;
    int *d_count;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));
    
    cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动 kernel
    kernel_copy_positive<<<1, 256>>>(d_src, d_dst, d_count);
    cudaDeviceSynchronize();
    
    // 拷贝结果回主机
    cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    int h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool success = true;
    int verified_count = 0;
    
    for (int i = 0; i < N; ++i) {
        if (h_src[i] > 0.0f) {
            // 应该被拷贝
            if (h_dst[i] == h_src[i]) {
                verified_count++;
            } else {
                success = false;
                std::cout << "错误: dst[" << i << "] = " << h_dst[i] 
                          << ", expected " << h_src[i] << "\n";
            }
        } else {
            // 不应该被拷贝
            if (h_dst[i] != -999.0f) {
                success = false;
                std::cout << "错误: dst[" << i << "] = " << h_dst[i] 
                          << ", 不应该被修改\n";
            }
        }
    }
    
    if (success) {
        std::cout << "✓ 条件拷贝成功！\n";
        std::cout << "  拷贝了 " << h_count << " 个正数\n";
        std::cout << "  验证了 " << verified_count << " 个元素\n";
    }
    
    // 清理
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_count);
}

//
// 边界处理场景：只拷贝有效范围内的元素
//
__global__ void kernel_copy_with_boundary(float* src, float* dst, int M, int N, 
                                          int tile_m, int tile_n) {
    const int TILE_SIZE = 16;
    
    // 计算当前 block 的起始位置
    int block_m = blockIdx.x * TILE_SIZE;
    int block_n = blockIdx.y * TILE_SIZE;
    
    __shared__ float smem[TILE_SIZE * TILE_SIZE];
    __shared__ bool pred[TILE_SIZE * TILE_SIZE];
    
    // 计算当前线程的全局坐标
    int tid = threadIdx.x;
    int local_m = tid / TILE_SIZE;
    int local_n = tid % TILE_SIZE;
    int global_m = block_m + local_m;
    int global_n = block_n + local_n;
    
    // 生成谓词：检查是否在边界内
    bool in_bounds = (global_m < M) && (global_n < N);
    pred[tid] = in_bounds;
    
    // 源和目标地址
    int src_idx = global_m * N + global_n;
    int dst_idx = (block_m + local_m) * tile_n + (block_n + local_n);
    
    // 创建 Tensor（每个线程处理一个元素）
    auto src_tensor = make_tensor(make_gmem_ptr(&src[src_idx]), make_shape(Int<1>{}));
    auto smem_tensor = make_tensor(make_smem_ptr(&smem[tid]), make_shape(Int<1>{}));
    auto pred_tensor = make_tensor(&pred[tid], make_shape(Int<1>{}));
    
    // 条件拷贝（边界外填充 0）
    if (in_bounds) {
        copy(src_tensor, smem_tensor);
    } else {
        smem[tid] = 0.0f;  // 边界外填充 0
    }
    
    __syncthreads();
    
    // 写回全局内存
    if (tid < TILE_SIZE * TILE_SIZE && dst_idx < tile_m * tile_n) {
        dst[dst_idx] = smem[tid];
    }
}

void test_boundary_handling() {
    std::cout << "\n=== 测试 3: 边界处理 ===\n";
    
    const int M = 100;  // 非对齐大小
    const int N = 100;
    const int TILE_SIZE = 16;
    const int tile_m = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    const int tile_n = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    
    std::cout << "矩阵大小: " << M << "×" << N << "\n";
    std::cout << "对齐后大小: " << tile_m << "×" << tile_n << "\n";
    
    // 主机端数据
    float* h_src = new float[M * N];
    float* h_dst = new float[tile_m * tile_n];
    
    // 初始化源数据
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            h_src[i * N + j] = i * 10 + j;
        }
    }
    
    // 初始化目标数据为 -1
    for (int i = 0; i < tile_m * tile_n; ++i) {
        h_dst[i] = -1.0f;
    }
    
    // 设备端数据
    float *d_src, *d_dst;
    cudaMalloc(&d_src, M * N * sizeof(float));
    cudaMalloc(&d_dst, tile_m * tile_n * sizeof(float));
    
    cudaMemcpy(d_src, h_src, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, tile_m * tile_n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动 kernel
    dim3 grid((tile_m + TILE_SIZE - 1) / TILE_SIZE, 
              (tile_n + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE * TILE_SIZE);
    
    kernel_copy_with_boundary<<<grid, block>>>(d_src, d_dst, M, N, tile_m, tile_n);
    cudaDeviceSynchronize();
    
    // 拷贝结果回主机
    cudaMemcpy(h_dst, d_dst, tile_m * tile_n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    bool success = true;
    int valid_count = 0;
    int boundary_count = 0;
    
    for (int i = 0; i < tile_m; ++i) {
        for (int j = 0; j < tile_n; ++j) {
            int idx = i * tile_n + j;
            if (i < M && j < N) {
                // 有效区域
                float expected = i * 10 + j;
                if (h_dst[idx] != expected) {
                    if (success) {  // 只打印第一个错误
                        std::cout << "错误: dst[" << i << "," << j << "] = " 
                                  << h_dst[idx] << ", expected " << expected << "\n";
                    }
                    success = false;
                } else {
                    valid_count++;
                }
            } else {
                // 边界外区域（应该是 0）
                if (h_dst[idx] != 0.0f) {
                    if (success) {
                        std::cout << "错误: 边界外 dst[" << i << "," << j << "] = " 
                                  << h_dst[idx] << ", expected 0\n";
                    }
                    success = false;
                } else {
                    boundary_count++;
                }
            }
        }
    }
    
    if (success) {
        std::cout << "✓ 边界处理成功！\n";
        std::cout << "  有效元素: " << valid_count << "\n";
        std::cout << "  边界填充元素: " << boundary_count << "\n";
    }
    
    // 清理
    delete[] h_src;
    delete[] h_dst;
    cudaFree(d_src);
    cudaFree(d_dst);
}

int main() {
    std::cout << "CuTe Copy_if 条件拷贝示例\n";
    std::cout << "=========================\n";
    
    // 测试 1: 主机端条件拷贝
    test_host_copy_if();
    
    // 测试 2: 设备端条件拷贝
    test_device_copy_positive();
    
    // 测试 3: 边界处理
    test_boundary_handling();
    
    std::cout << "\n所有测试完成！\n";
    
    return 0;
}

