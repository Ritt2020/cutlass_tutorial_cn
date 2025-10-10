#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

// 辅助函数：打印分隔线和标题
void print_section(const char* title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const char* title) {
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '-') << "\n";
}

int main() {
    print_section("CuTe 布局代数 (Layout Algebra) 示例");

    // ========================================================================
    // 1. Coalesce（合并/简化）
    // ========================================================================
    print_section("1. Coalesce - 布局合并/简化");

    print_subsection("示例 1.1：基本合并");
    
    // 创建一个嵌套的布局
    auto layout1 = make_layout(make_shape(Int<2>{}, make_shape(Int<1>{}, Int<6>{})),
                               make_stride(Int<1>{}, make_stride(Int<6>{}, Int<2>{})));
    std::cout << "原始布局:\n";
    print("  Layout: "); print(layout1); print("\n");
    print("  Shape:  "); print(shape(layout1)); print("\n");
    print("  Stride: "); print(stride(layout1)); print("\n");
    print("  Size:   "); print(size(layout1)); print("\n");
    
    // 合并后
    auto coalesced1 = coalesce(layout1);
    std::cout << "\n合并后:\n";
    print("  Layout: "); print(coalesced1); print("\n");
    print("  Shape:  "); print(shape(coalesced1)); print("\n");
    print("  Stride: "); print(stride(coalesced1)); print("\n");
    std::cout << "  说明: 嵌套结构被简化为单一模式\n";

    print_subsection("示例 1.2：列主序自动合并");
    
    auto layout2 = make_layout(make_shape(Int<2>{}, Int<4>{}));  // 默认列主序
    std::cout << "原始 2×4 列主序布局:\n";
    print("  Layout: "); print(layout2); print("\n");
    
    auto coalesced2 = coalesce(layout2);
    std::cout << "\n合并后:\n";
    print("  Layout: "); print(coalesced2); print("\n");
    std::cout << "  说明: (_2,_4):(_1,_2) 合并为 _8:_1（连续模式）\n";

    print_subsection("示例 1.3：按模式合并（保持维度结构）");
    
    auto layout3 = make_layout(make_shape(Int<2>{}, make_shape(Int<1>{}, Int<6>{})),
                               make_stride(Int<1>{}, make_stride(Int<6>{}, Int<2>{})));
    std::cout << "原始布局:\n";
    print("  Layout: "); print(layout3); print("\n");
    
    // 按模式合并，保持 2 维结构
    auto coalesced3 = coalesce(layout3, Step<_1,_1>{});
    std::cout << "\n按模式合并后（保持 2 维）:\n";
    print("  Layout: "); print(coalesced3); print("\n");
    std::cout << "  说明: 每个维度内部合并，但保持 rank=2\n";

    print_subsection("示例 1.4：无法合并的情况");
    
    auto layout4 = make_layout(make_shape(Int<2>{}, Int<4>{}),
                               make_stride(Int<6>{}, Int<1>{}));  // 带 padding
    std::cout << "带 padding 的布局:\n";
    print("  Layout: "); print(layout4); print("\n");
    print("  Stride: "); print(stride(layout4)); print("\n");
    
    auto coalesced4 = coalesce(layout4);
    std::cout << "\n尝试合并后:\n";
    print("  Layout: "); print(coalesced4); print("\n");
    std::cout << "  说明: 由于 stride 不连续，无法合并\n";

    // ========================================================================
    // 2. Composition（组合）
    // ========================================================================
    print_section("2. Composition - 函数组合");

    print_subsection("示例 2.1：基本组合");
    
    auto A = make_layout(make_shape(Int<6>{}, Int<2>{}),
                        make_stride(Int<8>{}, Int<2>{}));
    auto B = make_layout(make_shape(Int<4>{}, Int<3>{}),
                        make_stride(Int<3>{}, Int<1>{}));
    
    std::cout << "布局 A:\n";
    print("  A = "); print(A); print("\n");
    print("  Shape:  "); print(shape(A)); print("\n");
    print("  Stride: "); print(stride(A)); print("\n");
    
    std::cout << "\n布局 B:\n";
    print("  B = "); print(B); print("\n");
    print("  Shape:  "); print(shape(B)); print("\n");
    print("  Stride: "); print(stride(B)); print("\n");
    
    // 组合 R = A ∘ B，即 R(x) = A(B(x))
    auto R = composition(A, B);
    std::cout << "\n组合结果 R = A ∘ B:\n";
    print("  R = "); print(R); print("\n");
    print("  Shape:  "); print(shape(R)); print("\n");
    print("  Stride: "); print(stride(R)); print("\n");
    
    std::cout << "\n验证组合 R(i) = A(B(i)):\n";
    for (int i = 0; i < 5; i++) {
        int b_val = B(i);
        int a_b_val = A(b_val);
        int r_val = R(i);
        printf("  R(%d) = A(B(%d)) = A(%d) = %d (验证: %d) %s\n",
               i, i, b_val, a_b_val, r_val, (a_b_val == r_val) ? "✓" : "✗");
    }

    print_subsection("示例 2.2：将 1-D 数据重塑为矩阵");
    
    // 24 个元素的线性数据
    auto data_layout = make_layout(make_shape(24));
    std::cout << "线性数据布局 (24 个元素):\n";
    print("  Layout: "); print(data_layout); print("\n");
    
    // 重塑为 4×6 矩阵
    auto matrix_shape = make_layout(make_shape(Int<4>{}, Int<6>{}));
    std::cout << "\n目标矩阵形状 (4×6):\n";
    print("  Layout: "); print(matrix_shape); print("\n");
    
    // 组合：线性坐标 -> 矩阵坐标 -> 内存索引
    auto reshaped = composition(matrix_shape, data_layout);
    std::cout << "\n重塑后的布局:\n";
    print("  Layout: "); print(reshaped); print("\n");
    // print_layout 需要 rank=2 的布局
    if constexpr (decltype(rank(reshaped))::value == 2) {
        print_layout(reshaped);
    } else {
        std::cout << "  (布局 rank != 2，无法使用 print_layout 可视化)\n";
    }

    print_subsection("示例 2.3：复杂重塑");
    
    auto src = make_layout(make_shape(Int<2>{}, Int<6>{}),
                          make_stride(Int<6>{}, Int<1>{}));  // 2×6 行主序
    auto dst_shape = make_shape(Int<3>{}, Int<4>{});         // 重塑为 3×4
    
    std::cout << "源布局 (2×6 行主序):\n";
    print("  Layout: "); print(src); print("\n");
    print_layout(src);
    
    std::cout << "\n目标形状 (3×4):\n";
    print("  Shape: "); print(dst_shape); print("\n");
    
    auto composed = composition(src, dst_shape);
    std::cout << "\n组合后:\n";
    print("  Layout: "); print(composed); print("\n");
    print_layout(composed);

    // ========================================================================
    // 3. Complement（补集）
    // ========================================================================
    print_section("3. Complement - 布局补集");

    print_subsection("示例 3.1：简单 1-D 补集");
    
    auto tile1d = make_layout(make_shape(Int<4>{}), make_stride(Int<1>{}));
    std::cout << "原始 tile (4 个元素):\n";
    print("  Tile: "); print(tile1d); print("\n");
    print("  Size: "); print(size(tile1d)); print("\n");
    
    // 在大小 12 下的补集
    auto comp1d = complement(tile1d, 12);
    std::cout << "\n在大小 12 下的补集:\n";
    print("  Complement: "); print(comp1d); print("\n");
    print("  Size: "); print(size(comp1d)); print("\n");
    std::cout << "  说明: tile 重复了 " << size(comp1d) << " 次\n";

    print_subsection("示例 3.2：2-D 补集");
    
    auto tile2d = make_layout(make_shape(Int<2>{}, Int<2>{}),
                             make_stride(Int<4>{}, Int<1>{}));
    std::cout << "2×2 tile:\n";
    print("  Tile: "); print(tile2d); print("\n");
    print("  Size: "); print(size(tile2d)); print("\n");
    print("  Cosize: "); print(cosize(tile2d)); print("\n");
    print_layout(tile2d);
    
    // 在大小 24 下的补集
    auto comp2d = complement(tile2d, 24);
    std::cout << "\n在大小 24 下的补集:\n";
    print("  Complement: "); print(comp2d); print("\n");
    print("  Size: "); print(size(comp2d)); print("\n");
    std::cout << "  说明: 描述了 tile 如何重复 " << size(comp2d) << " 次填满空间\n";

    print_subsection("示例 3.3：补集与乘积的关系");
    
    auto A_tile = make_layout(make_shape(Int<2>{}, Int<2>{}),
                             make_stride(Int<4>{}, Int<1>{}));
    auto B_repeat = make_layout(make_shape(Int<6>{}));
    
    std::cout << "Tile A:\n";
    print("  A: "); print(A_tile); print("\n");
    std::cout << "\n重复模式 B (6 次):\n";
    print("  B: "); print(B_repeat); print("\n");
    
    // 计算 A 在 size(A)*size(B) 下的补集
    int total_size = size(A_tile) * size(B_repeat);
    auto A_comp = complement(A_tile, total_size);
    std::cout << "\nA 的补集 (total_size=" << total_size << "):\n";
    print("  A*: "); print(A_comp); print("\n");
    std::cout << "  说明: A* 描述了 tile 的重复模式\n";

    // ========================================================================
    // 4. Logical Divide（逻辑除法/切分）
    // ========================================================================
    print_section("4. Logical Divide - 布局切分");

    print_subsection("示例 4.1：基本 1-D 切分");
    
    auto data1d = make_layout(make_shape(Int<2>{}, Int<2>{}),
                             make_stride(Int<4>{}, Int<1>{}));  // size=4, cosize=6
    auto tiler1d = make_layout(make_shape(Int<6>{}));
    
    std::cout << "数据布局:\n";
    print("  Data: "); print(data1d); print("\n");
    print("  Size: "); print(size(data1d)); print("\n");
    
    std::cout << "\n切分器:\n";
    print("  Tiler: "); print(tiler1d); print("\n");
    print("  Size: "); print(size(tiler1d)); print("\n");
    
    auto divided1d = logical_divide(data1d, tiler1d);
    std::cout << "\n切分结果:\n";
    print("  Result: "); print(divided1d); print("\n");
    print("  Shape: "); print(shape(divided1d)); print("\n");
    print("  Stride: "); print(stride(divided1d)); print("\n");
    std::cout << "\n结构:\n";
    print("  Mode-0 (tile): "); print(layout<0>(divided1d)); print("\n");
    print("  Mode-1 (repetition): "); print(layout<1>(divided1d)); print("\n");

    print_subsection("示例 4.2：2-D 矩阵切分");
    
    // 6×20 的矩阵
    auto matrix = make_layout(make_shape(Int<6>{}, Int<20>{}));
    // 切分为 2×5 的块
    auto tile_shape = make_shape(Int<2>{}, Int<5>{});
    
    std::cout << "矩阵布局 (6×20):\n";
    print("  Matrix: "); print(matrix); print("\n");
    std::cout << "\nTile 大小 (2×5):\n";
    print("  Tile: "); print(tile_shape); print("\n");
    
    auto divided2d = logical_divide(matrix, tile_shape);
    std::cout << "\n切分结果:\n";
    print("  Result: "); print(divided2d); print("\n");
    print("  Rank: "); print(rank(divided2d)); print("\n");
    std::cout << "  说明: 矩阵被分成 (6/2) × (20/5) = 3×4 = 12 个块\n";

    print_subsection("示例 4.3：Zipped Divide");
    
    auto layout_zipped = make_layout(make_shape(Int<6>{}, Int<20>{}));
    auto tile_zipped = make_shape(Int<2>{}, Int<5>{});
    
    auto result_zipped = zipped_divide(layout_zipped, tile_zipped);
    std::cout << "Zipped Divide 结果:\n";
    print("  Result: "); print(result_zipped); print("\n");
    print("  Shape: "); print(shape(result_zipped)); print("\n");
    std::cout << "  特点: 形式为 ((M,N), (TileM,TileN))\n";

    print_subsection("示例 4.4：Tiled Divide");
    
    auto layout_tiled = make_layout(make_shape(Int<6>{}, Int<20>{}));
    auto tile_tiled = make_shape(Int<2>{}, Int<5>{});
    
    auto result_tiled = tiled_divide(layout_tiled, tile_tiled);
    std::cout << "Tiled Divide 结果:\n";
    print("  Result: "); print(result_tiled); print("\n");
    print("  Shape: "); print(shape(result_tiled)); print("\n");
    std::cout << "  特点: 形式为 ((M,N), TileM, TileN)\n";

    print_subsection("示例 4.5：Flat Divide");
    
    auto layout_flat = make_layout(make_shape(Int<6>{}, Int<20>{}));
    auto tile_flat = make_shape(Int<2>{}, Int<5>{});
    
    auto result_flat = flat_divide(layout_flat, tile_flat);
    std::cout << "Flat Divide 结果:\n";
    print("  Result: "); print(result_flat); print("\n");
    print("  Shape: "); print(shape(result_flat)); print("\n");
    std::cout << "  特点: 形式为 (M, N, TileM, TileN) - 完全展开\n";

    // ========================================================================
    // 5. Logical Product（逻辑乘积）
    // ========================================================================
    print_section("5. Logical Product - 布局乘积");

    print_subsection("示例 5.1：基本 1-D 乘积");
    
    auto tile_prod = make_layout(make_shape(Int<2>{}, Int<2>{}),
                                make_stride(Int<4>{}, Int<1>{}));
    auto repeat_prod = make_layout(make_shape(Int<6>{}));
    
    std::cout << "Tile:\n";
    print("  Tile: "); print(tile_prod); print("\n");
    print("  Size: "); print(size(tile_prod)); print("\n");
    
    std::cout << "\n重复次数:\n";
    print("  Repeat: "); print(repeat_prod); print("\n");
    print("  Size: "); print(size(repeat_prod)); print("\n");
    
    auto product1d = logical_product(tile_prod, repeat_prod);
    std::cout << "\n乘积结果:\n";
    print("  Product: "); print(product1d); print("\n");
    print("  Size: "); print(size(product1d)); print("\n");
    std::cout << "  说明: tile 重复了 " << size(repeat_prod) << " 次\n";

    print_subsection("示例 5.2：Blocked Product（块状乘积）");
    
    // 2×5 的 tile
    auto tile_2x5 = make_layout(make_shape(Int<2>{}, Int<5>{}));
    // 重复成 3×4 的网格
    auto grid_3x4 = make_layout(make_shape(Int<3>{}, Int<4>{}));
    
    std::cout << "Tile (2×5):\n";
    print("  Tile: "); print(tile_2x5); print("\n");
    std::cout << "\nGrid (3×4):\n";
    print("  Grid: "); print(grid_3x4); print("\n");
    
    auto blocked_prod = blocked_product(tile_2x5, grid_3x4);
    std::cout << "\nBlocked Product 结果:\n";
    print("  Result: "); print(blocked_prod); print("\n");
    print("  Shape: "); print(shape(blocked_prod)); print("\n");
    std::cout << "\n可视化:\n";
    print_layout(blocked_prod);
    std::cout << "  说明: (2,3) × (5,4) = 6×20 矩阵\n";

    print_subsection("示例 5.3：Raked Product（交错乘积）");
    
    auto raked_prod = raked_product(tile_2x5, grid_3x4);
    std::cout << "Raked Product 结果:\n";
    print("  Result: "); print(raked_prod); print("\n");
    print("  Shape: "); print(shape(raked_prod)); print("\n");
    std::cout << "\n可视化:\n";
    print_layout(raked_prod);
    std::cout << "  说明: tile 元素交错排列\n";

    print_subsection("示例 5.4：Zipped Product");
    
    auto zipped_prod = zipped_product(tile_2x5, grid_3x4);
    std::cout << "Zipped Product 结果:\n";
    print("  Result: "); print(zipped_prod); print("\n");
    print("  Shape: "); print(shape(zipped_prod)); print("\n");
    std::cout << "  特点: 形式为 ((M,N), (TileM,TileN,...))\n";

    print_subsection("示例 5.5：Tiled Product");
    
    auto tiled_prod = tiled_product(tile_2x5, grid_3x4);
    std::cout << "Tiled Product 结果:\n";
    print("  Result: "); print(tiled_prod); print("\n");
    print("  Shape: "); print(shape(tiled_prod)); print("\n");
    std::cout << "  特点: 形式为 ((M,N), TileM, TileN, ...)\n";

    // ========================================================================
    // 6. 实际应用场景
    // ========================================================================
    print_section("6. 实际应用场景");

    print_subsection("场景 6.1：矩阵分块（Matrix Tiling）");
    
    // 128×128 的矩阵分成 16×16 的块
    auto large_matrix = make_layout(make_shape(Int<128>{}, Int<128>{}));
    auto block_16x16 = make_shape(Int<16>{}, Int<16>{});
    
    std::cout << "大矩阵 (128×128):\n";
    print("  Matrix: "); print(large_matrix); print("\n");
    std::cout << "\nBlock 大小 (16×16):\n";
    print("  Block: "); print(block_16x16); print("\n");
    
    auto matrix_tiled = tiled_divide(large_matrix, block_16x16);
    std::cout << "\n分块后:\n";
    print("  Tiled: "); print(matrix_tiled); print("\n");
    print("  Shape: "); print(shape(matrix_tiled)); print("\n");
    std::cout << "  说明: 分成 (128/16) × (128/16) = 8×8 = 64 个块\n";

    print_subsection("场景 6.2：线程分区（Thread Partitioning）");
    
    // 128 个元素的数据
    auto data_128 = make_layout(make_shape(Int<128>{}));
    // 32 个线程（一个 warp）
    auto threads_32 = make_layout(make_shape(Int<32>{}));
    
    std::cout << "数据 (128 个元素):\n";
    print("  Data: "); print(data_128); print("\n");
    std::cout << "\n线程数 (32 个线程):\n";
    print("  Threads: "); print(threads_32); print("\n");
    
    auto per_thread = logical_divide(data_128, threads_32);
    std::cout << "\n每个线程的数据:\n";
    print("  Per-thread: "); print(per_thread); print("\n");
    std::cout << "  说明: 每个线程处理 " << size(layout<0>(per_thread)) << " 个元素\n";

    print_subsection("场景 6.3：Tensor Core 风格布局");
    
    // 创建适合 Tensor Core 的复杂布局
    auto tc_layout = make_layout(
        make_shape(make_shape(Int<16>{}, Int<16>{}),
                   make_shape(Int<8>{}, Int<8>{})),
        make_stride(make_stride(Int<1>{}, Int<16>{}),
                    make_stride(Int<256>{}, Int<4096>{}))
    );
    
    std::cout << "Tensor Core 风格布局:\n";
    print("  Layout: "); print(tc_layout); print("\n");
    print("  Shape: "); print(shape(tc_layout)); print("\n");
    print("  Stride: "); print(stride(tc_layout)); print("\n");
    print("  Size: "); print(size(tc_layout)); print("\n");
    print("  Cosize: "); print(cosize(tc_layout)); print("\n");
    std::cout << "  说明: 这种布局优化了 Tensor Core 的访问模式\n";

    print_subsection("场景 6.4：数据重排（行主序 ↔ 列主序）");
    
    constexpr int M = 8;
    constexpr int N = 12;
    
    // 行主序
    auto row_major = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                make_stride(Int<N>{}, Int<1>{}));
    std::cout << "行主序布局 (8×12):\n";
    print("  Row-major: "); print(row_major); print("\n");
    print_layout(row_major);
    
    // 列主序
    auto col_major = make_layout(make_shape(Int<M>{}, Int<N>{}),
                                make_stride(Int<1>{}, Int<M>{}));
    std::cout << "\n列主序布局 (8×12):\n";
    print("  Col-major: "); print(col_major); print("\n");
    print_layout(col_major);
    
    std::cout << "\n说明: 通过不同的 stride 实现不同的内存排列\n";

    // ========================================================================
    // 7. 组合使用示例
    // ========================================================================
    print_section("7. 组合使用示例");

    print_subsection("示例 7.1：矩阵分块 + 线程分区");
    
    // 64×64 矩阵
    auto gemm_matrix = make_layout(make_shape(Int<64>{}, Int<64>{}));
    // 分成 8×8 的块
    auto gemm_tile = make_shape(Int<8>{}, Int<8>{});
    
    std::cout << "步骤 1: 矩阵 64×64\n";
    print("  Matrix: "); print(gemm_matrix); print("\n");
    
    std::cout << "\n步骤 2: 分成 8×8 的块\n";
    auto gemm_tiled = tiled_divide(gemm_matrix, gemm_tile);
    print("  Tiled: "); print(gemm_tiled); print("\n");
    std::cout << "  说明: 分成 (64/8) × (64/8) = 8×8 = 64 个块\n";
    
    std::cout << "\n步骤 3: 每个块内的 64 个元素可以进一步分给多个线程\n";
    std::cout << "  例如 16 个线程，每个线程处理: 64/16 = 4 个元素\n";

    print_subsection("示例 7.2：先合并后切分");
    
    // 创建一个复杂但可简化的布局
    auto complex = make_layout(make_shape(Int<2>{}, Int<4>{}, Int<8>{}));
    std::cout << "复杂布局 (2×4×8):\n";
    print("  Complex: "); print(complex); print("\n");
    
    // 先合并
    auto simplified = coalesce(complex);
    std::cout << "\n合并后:\n";
    print("  Simplified: "); print(simplified); print("\n");
    
    // 再切分
    auto tile_simple = make_shape(Int<8>{});
    auto divided_simple = logical_divide(simplified, tile_simple);
    std::cout << "\n切分为 8 元素的块:\n";
    print("  Divided: "); print(divided_simple); print("\n");
    std::cout << "  说明: 64 个元素分成 8 个块，每块 8 个元素\n";

    print_subsection("示例 7.3：组合 + 乘积");
    
    // 定义一个 4×4 的 tile
    auto base_tile = make_layout(make_shape(Int<4>{}, Int<4>{}));
    // 重复成 2×3 的网格
    auto repeat_grid = make_layout(make_shape(Int<2>{}, Int<3>{}));
    
    std::cout << "Base tile (4×4):\n";
    print("  Tile: "); print(base_tile); print("\n");
    std::cout << "\nRepeat grid (2×3):\n";
    print("  Grid: "); print(repeat_grid); print("\n");
    
    auto repeated = blocked_product(base_tile, repeat_grid);
    std::cout << "\nRepeated layout (8×12):\n";
    print("  Repeated: "); print(repeated); print("\n");
    print("  Shape: "); print(shape(repeated)); print("\n");
    std::cout << "\n可视化（部分）:\n";
    print_layout(repeated);

    // ========================================================================
    // 8. 性能对比
    // ========================================================================
    print_section("8. 性能对比：静态 vs 动态");

    print_subsection("静态大小（编译时优化）");
    
    auto static_layout = make_layout(make_shape(Int<128>{}, Int<128>{}));
    std::cout << "静态布局 (128×128):\n";
    print("  Layout: "); print(static_layout); print("\n");
    std::cout << "  优点: 编译时已知，可以完全优化\n";
    std::cout << "  size: " << size(static_layout) << " (编译时常量)\n";

    print_subsection("动态大小（运行时计算）");
    
    int n = 128;
    auto dynamic_layout = make_layout(make_shape(n, n));
    std::cout << "动态布局 (" << n << "×" << n << "):\n";
    print("  Layout: "); print(dynamic_layout); print("\n");
    std::cout << "  缺点: 运行时计算，优化受限\n";
    std::cout << "  size: " << size(dynamic_layout) << " (运行时计算)\n";

    std::cout << "\n建议: 尽可能使用静态大小以获得最佳性能\n";

    // ========================================================================
    // 总结
    // ========================================================================
    print_section("总结");

    std::cout << R"(
CuTe 布局代数提供了强大的工具来操作布局：

1. Coalesce: 简化布局，减少模式数量
   - 不改变函数行为
   - 优化布局表示

2. Composition: 函数组合，将两个布局串联
   - R(x) = A(B(x))
   - 实现数据转换和重塑

3. Complement: 计算布局的"剩余空间"
   - 描述重复模式
   - size(A) × size(A*) = N

4. Logical Divide: 将布局按另一个布局切分
   - blocked: 块状分块
   - raked: 交错分块
   - zipped/tiled/flat: 不同的结果组织

5. Logical Product: 按另一个布局重复当前布局
   - 创建重复模式
   - 与 divide 互补

最佳实践：
✓ 优先使用静态大小（Int<N>{}）
✓ 使用 coalesce 简化复杂布局
✓ 选择合适的分块策略（blocked/raked）
✓ 使用可视化工具调试（print_layout/print_latex）
✓ 验证 size 和 cosize 属性
)";

    // ========================================================================
    // 实践练习
    // ========================================================================
    print_section("实践练习");

    // ------------------------------------------------------------------------
    // 任务 1：简化复杂布局
    // ------------------------------------------------------------------------
    print_subsection("任务 1：简化复杂布局");
    
    auto ex1_layout = make_layout(
        make_shape(Int<4>{}, make_shape(Int<2>{}, Int<3>{})),
        make_stride(Int<1>{}, make_stride(Int<4>{}, Int<8>{}))
    );
    
    std::cout << "原始布局:\n";
    print("  Layout: "); print(ex1_layout); print("\n");
    print("  Size:   "); print(size(ex1_layout)); print("\n");
    
    auto ex1_simplified = coalesce(ex1_layout);
    std::cout << "简化后:\n";
    print("  Layout: "); print(ex1_simplified); print("\n");
    print("  Size:   "); print(size(ex1_simplified)); print("\n");
    
    bool ex1_check = (size(ex1_layout) == size(ex1_simplified)) && (size(ex1_layout) == 24);
    std::cout << "验证: " << (ex1_check ? "✅ 通过" : "❌ 失败") << "\n";

    // ------------------------------------------------------------------------
    // 任务 2：矩阵格式转换
    // ------------------------------------------------------------------------
    print_subsection("任务 2：矩阵格式转换（行主序→列主序）");
    
    auto ex2_row_major = make_layout(
        make_shape(Int<8>{}, Int<12>{}),
        make_stride(Int<12>{}, Int<1>{})
    );
    
    auto ex2_col_major = make_layout(
        make_shape(Int<8>{}, Int<12>{}),
        make_stride(Int<1>{}, Int<8>{})
    );
    
    std::cout << "行主序布局:\n";
    print("  Layout: "); print(ex2_row_major); print("\n");
    
    std::cout << "列主序布局:\n";
    print("  Layout: "); print(ex2_col_major); print("\n");
    
    auto ex2_transformed = composition(ex2_col_major, ex2_row_major);
    std::cout << "转换后:\n";
    print("  Layout: "); print(ex2_transformed); print("\n");
    
    bool ex2_check = (size(ex2_transformed) == 96);
    std::cout << "验证: " << (ex2_check ? "✅ 通过" : "❌ 失败") << "\n";

    // ------------------------------------------------------------------------
    // 任务 3：GEMM 矩阵分块
    // ------------------------------------------------------------------------
    print_subsection("任务 3：GEMM 矩阵分块");
    
    auto ex3_matrix = make_layout(make_shape(Int<128>{}, Int<128>{}));
    
    std::cout << "原始矩阵 (128×128):\n";
    print("  Size: "); print(size(ex3_matrix)); print("\n");
    
    // 第一级：分成 32×32 的块
    auto ex3_block_tile = make_shape(Int<32>{}, Int<32>{});
    auto ex3_level1 = tiled_divide(ex3_matrix, ex3_block_tile);
    
    std::cout << "第一级分块（32×32 的 Block）:\n";
    print("  Layout: "); print(ex3_level1); print("\n");
    print("  Size:   "); print(size(ex3_level1)); print("\n");
    
    // 第二级：Block 内分成 4×4 的小块
    auto ex3_thread_tile = make_shape(Int<4>{}, Int<4>{});
    auto ex3_block_32x32 = make_layout(make_shape(Int<32>{}, Int<32>{}));
    auto ex3_level2 = tiled_divide(ex3_block_32x32, ex3_thread_tile);
    
    std::cout << "第二级分块（4×4 的 Thread 块）:\n";
    print("  Layout: "); print(ex3_level2); print("\n");
    print("  Size:   "); print(size(ex3_level2)); print("\n");
    
    bool ex3_check1 = (size(ex3_matrix) == 16384);
    bool ex3_check2 = (size(ex3_level1) == 16384);
    bool ex3_check3 = (size(ex3_level2) == 1024);
    std::cout << "验证: " << (ex3_check1 && ex3_check2 && ex3_check3 ? "✅ 通过" : "❌ 失败") << "\n";

    // ------------------------------------------------------------------------
    // 任务 4：Tile 重复模式
    // ------------------------------------------------------------------------
    print_subsection("任务 4：Tile 重复模式");
    
    auto ex4_tile = make_layout(make_shape(Int<2>{}, Int<5>{}));
    auto ex4_grid = make_layout(make_shape(Int<3>{}, Int<4>{}));
    
    std::cout << "原始 Tile (2×5):\n";
    print("  Tile: "); print(ex4_tile); print("\n");
    
    std::cout << "重复模式 (3×4):\n";
    print("  Grid: "); print(ex4_grid); print("\n");
    
    auto ex4_result = blocked_product(ex4_tile, ex4_grid);
    
    std::cout << "重复后的布局:\n";
    print("  Result: "); print(ex4_result); print("\n");
    print("  Shape:  "); print(shape(ex4_result)); print("\n");
    print("  Size:   "); print(size(ex4_result)); print("\n");
    
    bool ex4_check = (size(ex4_result) == 120);
    std::cout << "验证: " << (ex4_check ? "✅ 通过" : "❌ 失败") << "\n";

    print_section("所有任务完成！");

    return 0;
}

