//
// Copy_Atom 显式控制拷贝示例（需要 SM80+）
// 演示如何显式使用 cp.async.ca (cache all levels) 指令
//

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/arch/copy_sm80.hpp>

#include <cuda_runtime.h>
#include <iostream>

using namespace cute;

// 使用 __CUDA_ARCH__ 在设备代码中检查 SM80 支持
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)) || !defined(__CUDA_ARCH__)
#define SM80_ENABLED
#endif

#ifdef SM80_ENABLED

//
// 显式使用 cp.async.ca 拷贝 float 数据
//
__global__ void kernel_copy_with_cachealways(float* gmem, float* result) {
    __shared__ float smem[128];
    
    // 显式创建 Copy_Atom，指定使用 cp.async.ca
    // uint128_t 表示 16 字节（128 位）的拷贝
    using CopyOp = SM80_CP_ASYNC_CACHEALWAYS<uint128_t>;
    Copy_Atom<Copy_Traits<CopyOp>, float> copy_atom;
    
    // 32 个线程，每个线程拷贝 1 个 uint128_t（4 个 float）
    if (threadIdx.x < 32) {
        int offset = threadIdx.x * 4;
        // 直接创建每个线程的连续张量
        auto thr_g = make_tensor(make_gmem_ptr(&gmem[offset]), make_shape(Int<4>{}));
        auto thr_s = make_tensor(make_smem_ptr(&smem[offset]), make_shape(Int<4>{}));
        
        // 使用显式的 copy_atom 进行拷贝
        copy(copy_atom, thr_g, thr_s);
    }
    
    // 等待异步拷贝完成
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();
    
    // 将结果写回全局内存（每个线程写回 4 个元素）
    if (threadIdx.x < 32) {
        int offset = threadIdx.x * 4;
        for (int i = 0; i < 4; ++i) {
            result[offset + i] = smem[offset + i];
        }
    }
}

void test_copy_cachealways() {
    std::cout << "\n=== 使用 cp.async.ca 显式拷贝 float 数据 ===\n";
    
    const int N = 128;
    
    // 主机端数据
    float* h_input = new float[N];
    float* h_output = new float[N];
    
    // 初始化输入
    for (int i = 0; i < N; ++i) {
        h_input[i] = i * 1.5f;
    }
    
    // 设备端数据
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动 kernel（32 个线程）
    kernel_copy_with_cachealways<<<1, 32>>>(d_input, d_output);
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
        std::cout << "✓ cp.async.ca 拷贝成功！\n";
        std::cout << "  - 32 个线程，每个线程拷贝 4 个 float (16 字节)\n";
        std::cout << "  - 使用 uint128_t Copy_Atom 进行向量化拷贝\n";
        std::cout << "  - cache all levels 策略适用于会多次访问的数据\n";
    }
    
    // 清理
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
}

#endif // SM80_ENABLED

int main() {
    std::cout << "CuTe Copy_Atom 显式控制拷贝示例\n";
    std::cout << "==================================\n";
    
    // 检查 GPU 架构
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    
#ifdef SM80_ENABLED
    if (prop.major >= 8) {
        std::cout << "✓ 支持 SM80+ cp.async 指令\n";
        
        // 运行测试
        test_copy_cachealways();
        
        std::cout << "\n测试完成！\n";
    } else {
        std::cout << "✗ 需要 SM80+ (Ampere) 架构才能运行此示例\n";
        std::cout << "  当前 GPU 是 SM" << prop.major << prop.minor << "\n";
        return 1;
    }
#else
    std::cout << "✗ 编译时未启用 SM80 支持\n";
    std::cout << "  请使用 -arch=sm_80 或更高版本编译\n";
    return 1;
#endif
    
    return 0;
}

