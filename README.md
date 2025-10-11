# 中文 CUTLASS 教程

## 前置要求

本教程假设读者已经具备以下基础知识：

- **CUDA 编程基础**：熟悉 CUDA 核心概念（kernel、thread、block、grid、shared memory 等）
- **C++ 基础**：了解 C++ 11/17 特性，特别是模板（templates）的使用
- **线性代数基础**：理解矩阵乘法、向量运算等基本概念
- **GPU 架构基础**：对 GPU 内存层次结构有基本了解

如果你还不熟悉 CUDA，建议先学习 CUDA 基础课程。

## 目标架构

本教程以 Hopper(sm_90) 架构为目标架构进行。如果你的架构低于 sm_90，则你可以进行跳过以下部分，但强烈推荐也进行相关学习：
1. 第 7 节：TMA 拷贝部分
2. 第 8 节：WGMMA 部分

## 如何使用本教程

本教程采用**理论 + 实践**的学习方式，建议按照以下路径学习：

### 学习路径

1. **阅读文档**
   - 仔细阅读每一节的文档内容
   - 理解核心概念和示例代码
   - 文档中包含详细的原理说明和代码示例

2. **独立实践**
   - 根据文档中的"实践任务"部分，尝试独立完成练习
   - 不要急于查看参考代码，先自己思考和实现
   - 遇到困难时，可以回顾文档中的相关章节

3. **对照参考代码**
   - 完成练习后，查看 `code/` 目录下的对应代码
   - 对比自己的实现和参考代码的异同
   - 理解参考代码中的优化技巧和最佳实践

### 代码组织

每个章节都有对应的代码示例：
```
cutlass_tutorial_cn/
├── doc/              # 文档目录
│   ├── 01.安装配置.md
│   ├── 02.简单矩阵乘 GEMM.md
│   ├── 03.CuTe_Layout.md
│   └── ...
└── code/             # 代码目录
    ├── 02_simple_gemm/
    ├── 03_cute_layout/
    ├── 04_cute_layout_algebra/
    └── ...
```

## 目录

### 基础教程
1. [安装配置](doc/01.安装配置.md)
2. [简单矩阵乘 GEMM](doc/02.简单矩阵乘%20GEMM.md)
3. [CuTe Layout](doc/03.CuTe_Layout.md)
4. [CuTe Layout Algebra](doc/04.CuTe_Layout_Algebra.md)
5. [CuTe Tensor](doc/05.CuTe_Tensor.md)
6. [CuTe Copy](doc/06.CuTe_Copy.md)
7. [CuTe TMA Copy](doc/06.CuTe_TMA.md) TODO
8. [CuTe MMA](doc/07.CuTe_MMA.md) TODO
9. [Cutlass Pipeline](doc/08.Cutlass_Pipeline.md) TODO

### 附录
- [附录1. 数据类型](doc/附录1.%20数据类型.md)
- [附录2. 矩阵布局](doc/附录2.%20矩阵布局.md)

## 参考资料

以下是学习 CUTLASS 和 CuTe 的官方资源，强烈推荐配合本教程一起阅读：

### 官方文档
- [CUTLASS 官方文档](https://docs.nvidia.com/cutlass/index.html) - NVIDIA 官方维护的完整文档，包含 API 参考和架构说明
- [CUTLASS GitHub Wiki](https://github.com/NVIDIA/cutlass/wiki/Documentation) - GitHub 上的文档 Wiki，包含更多实践示例

### 博客文章
- [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/) - NVIDIA 开发者博客的入门介绍

### 代码仓库
- [CUTLASS GitHub Repository](https://github.com/NVIDIA/cutlass) - CUTLASS 的源代码仓库，包含丰富的示例代码

## 贡献

欢迎通过 Issue 或 Pull Request 为本教程做出贡献！
