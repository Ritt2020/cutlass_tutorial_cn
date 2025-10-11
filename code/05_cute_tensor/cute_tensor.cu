/***************************************************************************************************
 * CuTe Tensor 示例代码
 * 
 * 本文件演示 CuTe Tensor 的核心概念和操作：
 * 1. Tensor 的创建和访问
 * 2. 内存标记（make_gmem_ptr / make_smem_ptr）
 * 3. make_tensor_like 的使用
 * 4. Tensor 的切片（Slice）
 * 5. Tensor 的分区（Partition）
 * 6. Tensor 的算法（Copy、Fill）
 * 7. 实际应用场景
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>

// CuTe 头文件
#include "cute/tensor.hpp"

using namespace cute;

/***************************************************************************************************
 * 工具函数
 **************************************************************************************************/

// 打印分隔线
void print_separator(const char* title) {
    printf("\n");
    printf("========================================\n");
    printf("  %s\n", title);
    printf("========================================\n");
}

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/***************************************************************************************************
 * 1. Tensor 基础：创建和访问
 **************************************************************************************************/

// 主机端示例：创建和访问 Tensor
void test_tensor_basics() {
    print_separator("1. Tensor 基础：创建和访问");

    // 分配内存
    float data[24];
    
    // 方式 1：从指针和形状创建（默认列主序）
    auto tensor1 = make_tensor(&data[0], make_shape(Int<4>{}, Int<6>{}));
    
    printf("\n方式 1：默认列主序\n");
    print("  Shape:  "); print(shape(tensor1)); print("\n");
    print("  Stride: "); print(stride(tensor1)); print("\n");
    print("  Size:   "); print(size(tensor1)); print("\n");
    
    // 初始化：tensor(i,j) = i*10 + j
    for (int i = 0; i < size<0>(tensor1); ++i) {
        for (int j = 0; j < size<1>(tensor1); ++j) {
            tensor1(i, j) = i * 10.0f + j;
        }
    }
    
    // 打印部分元素
    printf("\n  访问示例：\n");
    printf("    tensor(0, 0) = %.1f\n", tensor1(0, 0));
    printf("    tensor(2, 3) = %.1f\n", tensor1(2, 3));
    printf("    tensor(3, 5) = %.1f\n", tensor1(3, 5));
    
    // 方式 2：指定步长（行主序）
    auto tensor2 = make_tensor(&data[0], 
                              make_shape(Int<4>{}, Int<6>{}),
                              make_stride(Int<6>{}, Int<1>{}));
    
    printf("\n方式 2：行主序\n");
    print("  Shape:  "); print(shape(tensor2)); print("\n");
    print("  Stride: "); print(stride(tensor2)); print("\n");
    
    // 方式 3：嵌套形状
    auto tensor3 = make_tensor(&data[0],
                              make_shape(Int<4>{}, make_shape(Int<2>{}, Int<3>{})));
    
    printf("\n方式 3：嵌套形状\n");
    print("  Shape:  "); print(shape(tensor3)); print("\n");
    print("  Stride: "); print(stride(tensor3)); print("\n");
    print("  Rank:   "); print(rank(tensor3)); print("\n");
    print("  Depth:  "); print(depth(tensor3)); print("\n");
    
    // 多种坐标形式访问同一元素
    printf("\n  多种坐标访问同一元素：\n");
    printf("    tensor3(5) = %.1f (一维索引)\n", tensor3(5));
    printf("    tensor3(1, 1) = %.1f (扁平坐标)\n", tensor3(1, 1));
    auto coord = make_coord(1, make_coord(1, 0));
    printf("    tensor3((1,(1,0))) = %.1f (层次化坐标)\n", tensor3(coord));
    printf("  （这三种方式访问的是同一个元素）\n");
}

/***************************************************************************************************
 * 2. make_tensor_like：创建形状相同的 Tensor
 **************************************************************************************************/

void test_make_tensor_like() {
    print_separator("2. make_tensor_like：创建形状相同的 Tensor");
    
    // 场景 1：创建与子视图相同形状的 Tensor
    printf("\n场景 1：创建与矩阵某一列形状相同的 Tensor\n");
    float data[8 * 16];
    auto matrix = make_tensor(&data[0], make_shape(Int<8>{}, Int<16>{}));
    
    // 初始化矩阵
    for (int i = 0; i < size(matrix); ++i) {
        matrix(i) = i;
    }
    
    printf("  矩阵 shape: "); print(shape(matrix)); printf("\n");
    
    // 提取第一列
    auto column = matrix(_, 0);
    printf("  第一列 shape: "); print(shape(column)); printf("\n");
    
    // 使用 make_tensor_like 创建形状相同的 Tensor
    // make_tensor_like 会自动分配存储空间（在主机端是数组，在设备端是寄存器）
    auto buffer_tensor = make_tensor_like(column);
    printf("  缓冲区 shape: "); print(shape(buffer_tensor)); printf("\n");
    
    // 拷贝数据验证
    for (int i = 0; i < size(column); ++i) {
        buffer_tensor(i) = column(i);
    }
    printf("  验证拷贝: 前 4 个元素 = ");
    for (int i = 0; i < 4; ++i) {
        printf("%.0f ", buffer_tensor(i));
    }
    printf("\n");
    
    // 场景 2：处理分块数据
    printf("\n场景 2：处理分块数据\n");
    auto tiled = zipped_divide(matrix, make_shape(Int<4>{}, Int<8>{}));
    printf("  Tiled shape: "); print(shape(tiled)); printf("\n");
    
    // 创建与一个 tile 形状相同的缓冲区
    // make_tensor_like 自动分配存储并匹配形状
    auto tile = tiled(make_coord(_,_), make_coord(0,0));
    auto tile_tensor = make_tensor_like(tile);
    printf("  单个 Tile shape: "); print(shape(tile)); printf("\n");
    printf("  Tile 缓冲区 shape: "); print(shape(tile_tensor)); printf("\n");
    printf("  Tile 缓冲区 size: %d\n", int(size(tile_tensor)));
}

/***************************************************************************************************
 * 3. Tensor 切片（Slice）
 **************************************************************************************************/

void test_tensor_slice() {
    print_separator("3. Tensor 切片（Slice）");
    
    // 创建 4×6 的矩阵
    float data[24];
    auto tensor = make_tensor(&data[0], make_shape(Int<4>{}, Int<6>{}));
    
    // 初始化
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = i;
    }
    
    printf("\n原始 Tensor: 4×6\n");
    print("  Shape: "); print(shape(tensor)); print("\n");
    
    // 切片 1：提取第 3 列（固定第二维）
    auto col3 = tensor(_, 3);
    printf("\n切片 1：第 3 列 tensor(_, 3)\n");
    print("  Shape: "); print(shape(col3)); print("\n");
    printf("  元素: ");
    for (int i = 0; i < size(col3); ++i) {
        printf("%.0f ", col3(i));
    }
    printf("\n");
    
    // 切片 2：提取第 2 行（固定第一维）
    auto row2 = tensor(2, _);
    printf("\n切片 2：第 2 行 tensor(2, _)\n");
    print("  Shape: "); print(shape(row2)); print("\n");
    printf("  元素: ");
    for (int i = 0; i < size(row2); ++i) {
        printf("%.0f ", row2(i));
    }
    printf("\n");
    
    // 切片 3：提取一个范围（2×3 子块，从 (1,2) 开始）
    printf("\n切片 3：2×3 子块，从 (1,2) 开始\n");
    printf("  元素:\n");
    for (int i = 0; i < 2; ++i) {
        printf("    ");
        for (int j = 0; j < 3; ++j) {
            printf("%3.0f ", tensor(1 + i, 2 + j));
        }
        printf("\n");
    }
}

/***************************************************************************************************
 * 4. Tensor 分区（Partition）- CUDA Kernel
 **************************************************************************************************/

// Kernel：演示 local_partition
__global__ void partition_kernel(float* data) {
    // 创建全局 Tensor（128 个元素）
    // 使用 make_gmem_ptr 标记全局内存，让 CuTe 选择最优的访问指令
    auto tensor = make_tensor(make_gmem_ptr(data), make_shape(Int<128>{}));
    
    // 定义线程布局（32 个线程）
    auto thread_layout = make_layout(Int<32>{});
    
    // 分区：每个线程得到一部分数据
    auto thread_tensor = local_partition(tensor, thread_layout, threadIdx.x);
    
    // 每个线程打印自己负责的范围
    if (threadIdx.x < 4) {  // 只让前 4 个线程打印
        printf("Thread %d: ", threadIdx.x);
        print("Shape = "); print(shape(thread_tensor));
        printf(", Elements = ");
        for (int i = 0; i < size(thread_tensor); ++i) {
            printf("%.0f ", thread_tensor(i));
        }
        printf("\n");
    }
}

void test_tensor_partition() {
    print_separator("4. Tensor 分区（Partition）");
    
    // 分配设备内存
    float* d_data;
    const int N = 128;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // 初始化数据
    float h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动 kernel
    printf("\n启动 32 个线程处理 128 个元素\n");
    printf("每个线程应该得到 4 个元素 (128 / 32 = 4)\n\n");
    partition_kernel<<<1, 32>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
}

/***************************************************************************************************
 * 5. Tensor 2D 分区 - CUDA Kernel
 **************************************************************************************************/

// Kernel：演示 2D Tensor 的分区
__global__ void partition_2d_kernel(float* data) {
    // 创建 16×16 的 Tensor
    auto tensor = make_tensor(make_gmem_ptr(data), make_shape(Int<16>{}, Int<16>{}));
    
    // 定义线程布局（64 个线程，排成 8×8）
    auto thread_layout = make_layout(make_shape(Int<8>{}, Int<8>{}));
    
    // 计算线程 ID（2D）
    int tid_x = threadIdx.x % 8;
    int tid_y = threadIdx.x / 8;
    int tid_2d = tid_y * 8 + tid_x;
    
    // 分区
    auto thread_tensor = local_partition(tensor, thread_layout, tid_2d);
    
    // 打印少数线程的信息
    if (threadIdx.x < 4) {
        printf("Thread %d: ", threadIdx.x);
        print("Shape = "); print(shape(thread_tensor));
        printf(", Size = %d\n", int(size(thread_tensor)));
    }
}

void test_tensor_partition_2d() {
    print_separator("5. Tensor 2D 分区");
    
    // 分配设备内存
    float* d_data;
    const int N = 16 * 16;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // 初始化数据
    float h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动 kernel
    printf("\n16×16 Tensor 分给 64 个线程 (8×8)\n");
    printf("每个线程应该得到 4 个元素 (256 / 64 = 4)\n\n");
    partition_2d_kernel<<<1, 64>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
}

/***************************************************************************************************
 * 6. Tensor 拷贝 - CUDA Kernel
 **************************************************************************************************/

// Kernel：演示 Tensor 拷贝（Global → Shared → Register）
__global__ void copy_kernel(float* src, float* dst) {
    // 全局内存 Tensor（使用 make_gmem_ptr 标记）
    auto g_src = make_tensor(make_gmem_ptr(src), make_shape(Int<256>{}));
    auto g_dst = make_tensor(make_gmem_ptr(dst), make_shape(Int<256>{}));
    
    // 共享内存 Tensor（使用 make_smem_ptr 标记）
    // 标记后，CuTe 可以自动使用优化的拷贝指令（如 cp.async）
    __shared__ float smem[256];
    auto s_tensor = make_tensor(make_smem_ptr(smem), make_shape(Int<256>{}));
    
    // 线程布局（32 个线程）
    auto thread_layout = make_layout(Int<32>{});
    
    // 分区：Global → Shared
    auto thr_g_src = local_partition(g_src, thread_layout, threadIdx.x);
    auto thr_s = local_partition(s_tensor, thread_layout, threadIdx.x);
    
    // 拷贝：Global → Shared（每个线程拷贝 8 个元素）
    #pragma unroll
    for (int i = 0; i < size(thr_g_src); ++i) {
        thr_s(i) = thr_g_src(i);
    }
    
    // 同步
    __syncthreads();
    
    // 寄存器存储
    float reg[8];
    auto r_tensor = make_tensor(&reg[0], make_shape(Int<8>{}));
    
    // 拷贝：Shared → Register
    #pragma unroll
    for (int i = 0; i < size(r_tensor); ++i) {
        r_tensor(i) = thr_s(i);
    }
    
    // 简单计算（验证数据正确）
    #pragma unroll
    for (int i = 0; i < size(r_tensor); ++i) {
        r_tensor(i) = r_tensor(i) * 2.0f;
    }
    
    // 拷贝回 Global
    auto thr_g_dst = local_partition(g_dst, thread_layout, threadIdx.x);
    #pragma unroll
    for (int i = 0; i < size(r_tensor); ++i) {
        thr_g_dst(i) = r_tensor(i);
    }
}

void test_tensor_copy() {
    print_separator("6. Tensor 拷贝（Global → Shared → Register → Global）");
    
    // 分配设备内存
    float* d_src;
    float* d_dst;
    const int N = 256;
    CUDA_CHECK(cudaMalloc(&d_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, N * sizeof(float)));
    
    // 初始化源数据
    float h_src[N];
    for (int i = 0; i < N; ++i) {
        h_src[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_src, h_src, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动 kernel
    printf("\n256 个元素，32 个线程并行处理\n");
    printf("流程：Global Memory → Shared Memory → Register → Global Memory\n");
    printf("每个元素 ×2\n\n");
    copy_kernel<<<1, 32>>>(d_src, d_dst);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 验证结果
    float h_dst[N];
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_dst[i] != h_src[i] * 2.0f) {
            correct = false;
            break;
        }
    }
    
    printf("验证结果: %s\n", correct ? "✓ 正确" : "✗ 错误");
    printf("前 10 个元素:\n");
    printf("  输入:  ");
    for (int i = 0; i < 10; ++i) printf("%.0f ", h_src[i]);
    printf("\n");
    printf("  输出:  ");
    for (int i = 0; i < 10; ++i) printf("%.0f ", h_dst[i]);
    printf("\n");
    printf("  预期:  ");
    for (int i = 0; i < 10; ++i) printf("%.0f ", h_src[i] * 2.0f);
    printf("\n");
    
    // 清理
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
}

/***************************************************************************************************
 * 7. Tiled Partition（平铺分区）- CUDA Kernel
 **************************************************************************************************/

// Kernel：演示 tiled partition
__global__ void tiled_partition_kernel(float* data) {
    // 创建 32×32 的 Tensor
    auto tensor = make_tensor(make_gmem_ptr(data), make_shape(Int<32>{}, Int<32>{}));
    
    // 定义 tile 大小（8×8）
    auto tile_shape = make_shape(Int<8>{}, Int<8>{});
    
    // 使用 zipped_divide 创建 tiled tensor
    auto tiled = zipped_divide(tensor, tile_shape);
    // 结果 shape: ((8,8), (4,4))
    // 前半部分 (8,8) 是单个 tile
    // 后半部分 (4,4) 是 tile 的网格（4×4 = 16 个 tile）
    
    if (threadIdx.x == 0) {
        printf("原始 Tensor shape: "); print(shape(tensor)); printf("\n");
        printf("Tiled Tensor shape: "); print(shape(tiled)); printf("\n");
        printf("单个 tile shape: "); print(shape<0>(tiled)); printf("\n");
        printf("Tile 网格 shape: "); print(shape<1>(tiled)); printf("\n");
    }
    
    // 现在可以分配 tile 给不同线程/block
    // 例如：每个 block 处理一个 tile
}

void test_tiled_partition() {
    print_separator("7. Tiled Partition（平铺分区）");
    
    // 分配设备内存
    float* d_data;
    const int N = 32 * 32;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // 初始化数据
    float h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动 kernel
    printf("\n32×32 Tensor 切分成 8×8 的 tile\n");
    printf("应该得到 4×4 = 16 个 tile\n\n");
    tiled_partition_kernel<<<1, 1>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
}

/***************************************************************************************************
 * 实践任务
 **************************************************************************************************/

// 任务 1：创建和访问 Tensor
void task1_create_and_access() {
    print_separator("任务 1：创建和访问 Tensor");
    
    printf("\n创建一个 8×12 的 Tensor，用 行号×10 + 列号 填充\n\n");
    
    float data[8 * 12];
    auto tensor = make_tensor(&data[0], make_shape(Int<8>{}, Int<12>{}));
    
    // 填充数据
    for (int i = 0; i < size<0>(tensor); ++i) {
        for (int j = 0; j < size<1>(tensor); ++j) {
            tensor(i, j) = i * 10.0f + j;
        }
    }
    
    // 验证
    printf("验证 tensor(2, 5):\n");
    printf("  实际值: %.0f\n", tensor(2, 5));
    printf("  预期值: %.0f\n", 2 * 10.0f + 5);
    printf("  结果: %s\n", tensor(2, 5) == 25.0f ? "✓ 正确" : "✗ 错误");
    
    // 打印部分元素
    printf("\n部分元素（前 4 行 6 列）:\n");
    for (int i = 0; i < 4; ++i) {
        printf("  ");
        for (int j = 0; j < 6; ++j) {
            printf("%4.0f ", tensor(i, j));
        }
        printf("\n");
    }
}

// 任务 2：Tensor 切片
void task2_tensor_slice() {
    print_separator("任务 2：Tensor 切片");
    
    printf("\n创建 16×16 的 Tensor 并提取切片\n\n");
    
    float data[16 * 16];
    auto tensor = make_tensor(&data[0], make_shape(Int<16>{}, Int<16>{}));
    
    // 初始化
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = i;
    }
    
    // 1. 第 8 行
    auto row8 = tensor(8, _);
    printf("1. 第 8 行:\n");
    print("   Shape: "); print(shape(row8)); printf("\n");
    printf("   前 8 个元素: ");
    for (int i = 0; i < 8; ++i) {
        printf("%.0f ", row8(i));
    }
    printf("\n");
    
    // 2. 第 5 列
    auto col5 = tensor(_, 5);
    printf("\n2. 第 5 列:\n");
    print("   Shape: "); print(shape(col5)); printf("\n");
    printf("   前 8 个元素: ");
    for (int i = 0; i < 8; ++i) {
        printf("%.0f ", col5(i));
    }
    printf("\n");
    
    // 3. 左上角 4×4 子块
    auto tile = local_tile(tensor, make_shape(Int<4>{}, Int<4>{}), make_coord(0, 0));
    printf("\n3. 左上角 4×4 子块:\n");
    print("   Shape: "); print(shape(tile)); printf("\n");
    printf("   元素:\n");
    for (int i = 0; i < size<0>(tile); ++i) {
        printf("     ");
        for (int j = 0; j < size<1>(tile); ++j) {
            printf("%4.0f ", tile(i, j));
        }
        printf("\n");
    }
}

// 任务 3：Tensor 分区（CUDA Kernel）
__global__ void task3_partition_kernel(float* data) {
    // 128 个元素
    auto tensor = make_tensor(make_gmem_ptr(data), make_shape(Int<128>{}));
    
    // 32 个线程
    auto thread_layout = make_layout(Int<32>{});
    auto thread_tensor = local_partition(tensor, thread_layout, threadIdx.x);
    
    // 使用顺序打印，避免输出混乱
    for (int tid = 0; tid < 8; ++tid) {
        if (threadIdx.x == tid) {
            printf("Thread %2d: 索引 [", threadIdx.x);
            for (int i = 0; i < size(thread_tensor); ++i) {
                // 计算全局索引
                int global_idx = threadIdx.x + i * 32;
                printf("%3d", global_idx);
                if (i < size(thread_tensor) - 1) printf(", ");
            }
            printf("] = 值 [");
            for (int i = 0; i < size(thread_tensor); ++i) {
                printf("%3.0f", thread_tensor(i));
                if (i < size(thread_tensor) - 1) printf(", ");
            }
            printf("]\n");
        }
        __syncthreads();
    }
}

void task3_partition() {
    print_separator("任务 3：Tensor 分区");
    
    printf("\n32 个线程处理 128 个元素\n");
    printf("每个线程应该得到 4 个元素 (128 / 32 = 4)\n");
    printf("只显示前 8 个线程的输出\n\n");
    
    // 分配设备内存
    float* d_data;
    const int N = 128;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    
    // 初始化
    float h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动 kernel
    task3_partition_kernel<<<1, 32>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
}

// 任务 4：使用 make_tensor_like 和内存标记
__global__ void task4_tensor_like_kernel(float* data, float* result) {
    // 创建 8×16 的全局内存 Tensor（使用 make_gmem_ptr 标记）
    auto gmem = make_tensor(make_gmem_ptr(data), make_shape(Int<8>{}, Int<16>{}));
    auto gmem_result = make_tensor(make_gmem_ptr(result), make_shape(Int<8>{}, Int<16>{}));
    
    // 使用 make_tensor_like 创建寄存器缓冲区（形状与一列相同）
    // 这里创建一个拥有数据的 Tensor（会分配寄存器）
    auto rmem = make_tensor_like(gmem(_, 0));
    
    if (threadIdx.x == 0) {
        printf("全局内存 Tensor shape: "); print(shape(gmem)); printf("\n");
        printf("一列的 shape: "); print(shape(gmem(_, 0))); printf("\n");
        printf("寄存器缓冲区 shape: "); print(shape(rmem)); printf("\n");
        printf("\n逐列处理（共 16 列）...\n\n");
    }
    
    // 逐列拷贝并处理数据
    for (int j = 0; j < size<1>(gmem); ++j) {
        // 拷贝第 j 列到寄存器
        #pragma unroll
        for (int i = 0; i < size(rmem); ++i) {
            rmem(i) = gmem(i, j);
        }
        
        // 在寄存器中处理数据（例如：平方）
        #pragma unroll
        for (int i = 0; i < size(rmem); ++i) {
            rmem(i) = rmem(i) * rmem(i);
        }
        
        // 写回结果
        #pragma unroll
        for (int i = 0; i < size(rmem); ++i) {
            gmem_result(i, j) = rmem(i);
        }
    }
}

void task4_tensor_like() {
    print_separator("任务 4：使用 make_tensor_like 和内存标记");
    
    printf("\n从全局内存逐列读取 8×16 矩阵到寄存器并处理\n");
    printf("操作：对每个元素求平方\n\n");
    
    // 分配设备内存
    float* d_data;
    float* d_result;
    const int M = 8;
    const int N = 16;
    CUDA_CHECK(cudaMalloc(&d_data, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, M * N * sizeof(float)));
    
    // 初始化输入数据
    float h_data[M * N];
    for (int i = 0; i < M * N; ++i) {
        h_data[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data, M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动 kernel（单线程演示）
    task4_tensor_like_kernel<<<1, 1>>>(d_data, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 验证结果
    float h_result[M * N];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_result[i] != h_data[i] * h_data[i]) {
            correct = false;
            break;
        }
    }
    
    printf("验证结果: %s\n", correct ? "✓ 正确" : "✗ 错误");
    printf("\n前 3 列结果:\n");
    for (int i = 0; i < M; ++i) {
        printf("  行 %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%6.0f ", h_result[i * N + j]);
        }
        printf("  (预期: ");
        for (int j = 0; j < 3; ++j) {
            printf("%6.0f ", h_data[i * N + j] * h_data[i * N + j]);
        }
        printf(")\n");
    }
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
}

/***************************************************************************************************
 * 主函数
 **************************************************************************************************/

int main() {
    printf("╔══════════════════════════════════════╗\n");
    printf("║   CuTe Tensor 示例代码               ║\n");
    printf("╚══════════════════════════════════════╝\n");
    
    // 基础示例
    test_tensor_basics();
    test_make_tensor_like();
    test_tensor_slice();
    
    // CUDA Kernel 示例
    test_tensor_partition();
    test_tensor_partition_2d();
    test_tensor_copy();
    test_tiled_partition();
    
    // 实践任务
    printf("\n\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║         实践任务                     ║\n");
    printf("╚══════════════════════════════════════╝\n");
    
    task1_create_and_access();
    task2_tensor_slice();
    task3_partition();
    task4_tensor_like();
    
    printf("\n\n");
    printf("╔══════════════════════════════════════╗\n");
    printf("║         所有测试完成！               ║\n");
    printf("╚══════════════════════════════════════╝\n");
    
    return 0;
}

