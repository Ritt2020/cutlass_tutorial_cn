// 说明：
// 本示例展示如何使用 CUTLASS 在设备端执行列主序(Column-Major)的矩阵乘法 GEMM：
//   C = alpha * A(m x k) * B(k x n) + beta * C(m x n)
//
// 关键点：
// - A、B、C 均为列主序存储（cutlass::layout::ColumnMajor）
// - 矩阵A使用高斯分布随机填充，矩阵B使用顺序填充
// - 使用 cutlass::gemm::device::Gemm 作为设备端内核
// - 同时使用 CPU GEMM API 和 GPU GEMM API 进行计算并比较结果
//
// 依赖：请确保 CUTLASS 头文件位于 /sharedir44-48/wanghy/cutlass/include/cutlass
// 编译示例（可能需要根据你的 CUDA/CMake 环境调整）：
//   nvcc -I/sharedir44-48/wanghy/cutlass/include -I/sharedir44-48/wanghy/cutlass/tools/util/include -O3 -arch=sm_80 \
//        /sharedir44-48/wanghy/cuda_playground/cutlass/test.cu -o test_gemm
// 运行：
//   ./test_gemm

// 标准库头文件
#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <chrono>

// 半精度GEMM内核所需的CUTLASS头文件
#include "cutlass/cutlass.h"           // CUTLASS核心头文件
#include "cutlass/core_io.h"           // 核心I/O功能
#include "cutlass/layout/matrix.h"     // 矩阵布局定义
#include "cutlass/gemm/device/gemm.h"  // 设备端GEMM内核

//
// CUTLASS实用工具头文件
//

// 定义operator<<()以将TensorView对象写入std::ostream
#include "cutlass/util/tensor_view_io.h"

// 定义cutlass::HostTensor<>模板类
#include "cutlass/util/host_tensor.h"

// 定义cutlass::half_t半精度类型
#include "cutlass/numeric_types.h"

// 定义device_memory::copy_device_to_device()函数
#include "cutlass/util/device_memory.h"

// 定义cutlass::reference::device::TensorFillRandomGaussian()函数
#include "cutlass/util/reference/device/tensor_fill.h"

// 定义cutlass::reference::host::TensorEquals()函数
#include "cutlass/util/reference/host/tensor_compare.h"

// 定义cutlass::reference::host::Gemm()函数
#include "cutlass/util/reference/host/gemm.h"

// 简单的 CUDA 错误检查宏
#define CUDA_CHECK(expr)                                                                    \
  do {                                                                                      \
    cudaError_t _err = (expr);                                                              \
    if (_err != cudaSuccess) {                                                              \
      std::cerr << "CUDA Error: " << cudaGetErrorString(_err) << " at " << __FILE__        \
                << ":" << __LINE__ << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                                              \
    }                                                                                       \
  } while (0)

/// 定义CUTLASS GEMM模板并启动半精度GEMM内核
/// 这个函数演示了如何使用CUTLASS进行半精度矩阵乘法计算
cudaError_t cutlass_hgemm_nn(
  int M,                                          // 矩阵A的行数，矩阵C的行数
  int N,                                          // 矩阵B的列数，矩阵C的列数
  int K,                                          // 矩阵A的列数，矩阵B的行数
  cutlass::half_t alpha,                         // 标量系数alpha（半精度）
  cutlass::half_t const *A,                      // 输入矩阵A（半精度）
  cutlass::layout::ColumnMajor::Stride::Index lda, // 矩阵A的leading dimension
  cutlass::half_t const *B,                      // 输入矩阵B（半精度）
  cutlass::layout::ColumnMajor::Stride::Index ldb, // 矩阵B的leading dimension
  cutlass::half_t beta,                          // 标量系数beta（半精度）
  cutlass::half_t *C,                            // 输入/输出矩阵C（半精度）
  cutlass::layout::ColumnMajor::Stride::Index ldc) { // 矩阵C的leading dimension

  // 定义半精度GEMM操作
  // 注意：所有矩阵都使用半精度类型和列主序布局
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // 矩阵A的元素类型（半精度）
    cutlass::layout::ColumnMajor,              // 矩阵A的布局（列主序）
    cutlass::half_t,                           // 矩阵B的元素类型（半精度）
    cutlass::layout::ColumnMajor,              // 矩阵B的布局（列主序）
    cutlass::half_t,                           // 输出矩阵的元素类型（半精度）
    cutlass::layout::ColumnMajor               // 输出矩阵的布局（列主序）
  >;

  // 创建GEMM操作符实例
  Gemm gemm_op;
  
  // 调用GEMM操作符，传入参数对象
  // 参数对象包含问题维度、矩阵引用和标量系数
  cutlass::Status status = gemm_op({
    {M, N, K},        // 问题维度：M×N×K
    {A, lda},         // 矩阵A的张量引用
    {B, ldb},         // 矩阵B的张量引用
    {C, ldc},         // 矩阵C的张量引用（输入）
    {C, ldc},         // 矩阵D的张量引用（输出，与C相同）
    {alpha, beta}     // 标量系数
  });

  // 检查操作是否成功
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;  // 返回未知CUDA错误
  }

  return cudaSuccess;  // 返回成功
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 在GPU设备内存中分配多个矩阵并调用半精度CUTLASS GEMM内核
/// 这个函数演示了CUTLASS实用工具的使用，包括HostTensor、随机初始化和结果验证
cudaError_t TestCutlassGemm(int M, int N, int K, cutlass::half_t alpha, cutlass::half_t beta) {
  cudaError_t result;

  //
  // 使用半精度主机端类型构造cutlass::HostTensor<>
  //
  // cutlass::HostTensor<>为列主序布局的rank=2张量在主机和设备上分配内存。
  // 提供显式同步方法来将张量复制到设备或主机。
  //

  // M×K的半精度矩阵A
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));

  // K×N的半精度矩阵B
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(cutlass::MatrixCoord(K, N));

  // M×N的半精度矩阵C_cutlass（CUTLASS计算结果）
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_cutlass(cutlass::MatrixCoord(M, N));

  // M×N的半精度矩阵C_reference（参考计算结果）
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> C_reference(cutlass::MatrixCoord(M, N));

  //
  // 使用小的随机整数初始化矩阵
  //

  // 任意RNG种子值。硬编码以确保确定性结果
  uint64_t seed = 2080;

  // 高斯随机分布参数
  cutlass::half_t mean = 0.0_hf;      // 均值
  cutlass::half_t stddev = 5.0_hf;    // 标准差

  // 指定二进制小数点右侧允许非零的位数
  // 值为"0"会将随机值截断为整数
  int bits_less_than_one = 0;

  // 使用设备端随机填充函数初始化矩阵A
  cutlass::reference::device::TensorFillRandomGaussian(
    A.device_view(),        // 矩阵A的设备视图
    seed,                   // 随机种子
    mean,                   // 均值
    stddev,                 // 标准差
    bits_less_than_one     // 精度控制
  );
  
  // 初始化矩阵B
  cutlass::reference::device::TensorFillRandomGaussian(
    B.device_view(),        // 矩阵A的设备视图
    seed,                   // 随机种子
    mean,                   // 均值
    stddev,                 // 标准差
    bits_less_than_one     // 精度控制
  );
  
  // 使用另一个种子初始化矩阵C_cutlass
  cutlass::reference::device::TensorFillRandomGaussian(
    C_cutlass.device_view(),
    seed * 1993,            // 另一个不同的种子
    mean,
    stddev,
    bits_less_than_one
  );


  // 将C_cutlass复制到C_reference，这样当beta != 0时GEMM计算是正确的
  cutlass::device_memory::copy_device_to_device(
    C_reference.device_data(),    // 目标设备内存
    C_cutlass.device_data(),      // 源设备内存
    C_cutlass.capacity());        // 复制的字节数

  // 将设备端视图复制到主机内存
  C_reference.sync_host();

  //
  // 启动CUTLASS GEMM内核
  //

  // 调用半精度GEMM函数进行计算
  result = cutlass_hgemm_nn(
    M,                        // 矩阵A的行数
    N,                        // 矩阵B的列数
    K,                        // 矩阵A的列数，矩阵B的行数
    alpha,                    // 标量系数alpha
    A.device_data(),          // 矩阵A的设备数据指针
    A.stride(0),              // 矩阵A的步长
    B.device_data(),          // 矩阵B的设备数据指针
    B.stride(0),              // 矩阵B的步长
    beta,                     // 标量系数beta
    C_cutlass.device_data(),  // 矩阵C的设备数据指针
    C_cutlass.stride(0)       // 矩阵C的步长
  );

  if (result != cudaSuccess) {
    return result;  // 如果GEMM失败，返回错误
  }

  //
  // 使用主机端参考实现验证结果
  //

  // A和B使用设备端过程初始化。此示例的意图是使用主机端参考GEMM，
  // 所以我们必须执行设备到主机的复制。
  A.sync_host();        // 将矩阵A从设备同步到主机
  B.sync_host();        // 将矩阵B从设备同步到主机

  // 将CUTLASS的GEMM结果复制到主机内存
  C_cutlass.sync_host();

  // 使用主机端GEMM参考实现计算参考结果
  cutlass::reference::host::Gemm<
    cutlass::half_t,                           // 矩阵A的元素类型
    cutlass::layout::ColumnMajor,              // 矩阵A的布局
    cutlass::half_t,                           // 矩阵B的元素类型
    cutlass::layout::ColumnMajor,              // 矩阵B的布局
    cutlass::half_t,                           // 输出矩阵的元素类型
    cutlass::layout::ColumnMajor,              // 输出矩阵的布局
    cutlass::half_t,                           // 内部累加类型
    cutlass::half_t                            // 标量类型
  > gemm_ref;

  // 调用主机端参考GEMM实现
  gemm_ref(
    {M, N, K},                          // 问题大小（类型：cutlass::gemm::GemmCoord）
    alpha,                              // alpha（类型：cutlass::half_t）
    A.host_ref(),                       // 矩阵A（类型：TensorRef<half_t, ColumnMajor>）
    B.host_ref(),                       // 矩阵B（类型：TensorRef<half_t, ColumnMajor>）
    beta,                               // beta（类型：cutlass::half_t）
    C_reference.host_ref()              // 矩阵C（类型：TensorRef<half_t, ColumnMajor>）
  );

  // 比较参考结果和计算结果
  if (!cutlass::reference::host::TensorEquals(
    C_reference.host_view(),    // 参考结果的主机视图
    C_cutlass.host_view())) {   // CUTLASS结果的主机视图

    char const *filename = "errors_01_cutlass_utilities.csv";

    std::cerr << "错误 - CUTLASS GEMM内核与参考实现不同。已将计算结果和参考结果写入 '" << filename << "'" << std::endl;

    //
    // 出错时，将C_cutlass和C_reference打印到std::cerr
    //
    // 注意，这些是半精度元素的矩阵，在主机内存中存储为cutlass::half_t类型的数组
    //

    std::ofstream file(filename);

    // CUTLASS GEMM内核的结果
    file << "\n\nCUTLASS =\n" << C_cutlass.host_view() << std::endl;

    // 参考计算的结果
    file << "\n\nReference =\n" << C_reference.host_view() << std::endl;

    // 返回错误代码
    return cudaErrorUnknown;
  }

  // 通过错误检查
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// cutlass_utilities示例的入口点
//
// 使用方法:
//
//   01_cutlass_utilities <M> <N> <K> <alpha> <beta>
//
// 参数说明:
//   M: 矩阵A的行数，矩阵C的行数
//   N: 矩阵B的列数，矩阵C的列数
//   K: 矩阵A的列数，矩阵B的行数
//   alpha: 标量系数alpha（半精度）
//   beta: 标量系数beta（半精度）
int main(int argc, const char *arg[]) {

  //
  // 此示例使用半精度，仅适用于计算能力5.3或更高的设备
  //

  cudaDeviceProp prop;
  cudaError_t result = cudaGetDeviceProperties(&prop, 0);
  
  if (result != cudaSuccess) {
    std::cerr << "查询设备属性失败，错误: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  // 检查设备计算能力是否支持半精度
  if (!(prop.major > 5 || (prop.major == 5 && prop.minor >= 3))) {
    std::cerr << "此示例使用半精度，仅适用于计算能力5.3或更高的设备。\n";
    std::cerr << "您正在使用计算能力为 " << prop.major << "." << prop.minor << " 的CUDA设备" << std::endl;
    return -1;
  }

  //
  // 解析命令行参数以获取GEMM维度和标量值
  //

  // GEMM问题维度：<M> <N> <K>，默认值为128x128x128
  int problem[3] = { 128, 128, 128 };

  // 解析前三个参数作为矩阵维度
  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // GEMM中的线性缩放因子。注意，这些是存储为cutlass::half_t的半精度值
  //
  // 超出IEEE FP16范围的值将溢出到无穷大或下溢到零
  //
  cutlass::half_t scalars[2] = { 1.0_hf, 0.0_hf };  // 默认alpha=1.0, beta=0.0

  // 解析第4和第5个参数作为标量系数
  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);

    ss >> scalars[i - 4];   // 词法转换到cutlass::half_t
  }

  //
  // 运行CUTLASS GEMM测试
  //

  result = TestCutlassGemm(
    problem[0],     // GEMM M维度
    problem[1],     // GEMM N维度
    problem[2],     // GEMM K维度
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  // 输出测试结果
  if (result == cudaSuccess) {
    std::cout << "测试通过。" << std::endl;
  } else {
    std::cout << "测试失败。" << std::endl;
  }

  // 退出程序
  return result == cudaSuccess ? 0 : -1;
}


