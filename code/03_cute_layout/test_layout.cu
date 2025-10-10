#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

// 辅助函数：打印分隔线和标题
void print_section(const char* title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

int main() {
    print_section("CuTe Layout 示例");

    // ============================================================
    // 1. 一维布局
    // ============================================================
    print_section("1. 一维布局");
    
    // 静态大小
    auto s8 = make_layout(Int<8>{});
    std::cout << "s8 (静态):\n";
    print("  Layout: "); print(s8); print("\n");
    print("  Shape:  "); print(shape(s8)); print("\n");
    print("  Stride: "); print(stride(s8)); print("\n");
    print("  Size:   "); print(size(s8)); print("\n");
    print("  Cosize: "); print(cosize(s8)); print("\n");
    print("\n");
    
    // 动态大小
    auto d8 = make_layout(8);
    std::cout << "d8 (动态):\n";
    print("  Layout: "); print(d8); print("\n");
    print("  Shape:  "); print(shape(d8)); print("\n");
    print("  Stride: "); print(stride(d8)); print("\n");
    print("  Size:   "); print(size(d8)); print("\n");
    print("  Cosize: "); print(cosize(d8)); print("\n");

    // ============================================================
    // 2. 二维布局 - 列主序（默认）
    // ============================================================
    print_section("2. 二维布局 - 列主序（默认）");
    
    auto s2xs4 = make_layout(make_shape(Int<2>{}, Int<4>{}));
    std::cout << "s2xs4 (2x4 静态，列主序):\n";
    print("  Layout: "); print(s2xs4); print("\n");
    print("  Shape:  "); print(shape(s2xs4)); print("\n");
    print("  Stride: "); print(stride(s2xs4)); print("\n");
    print("  Size:   "); print(size(s2xs4)); print("\n");
    print("  Cosize: "); print(cosize(s2xs4)); print("\n");
    print("  2D Layout:\n");
    print_layout(s2xs4);

    // ============================================================
    // 3. 二维布局 - 混合静态动态
    // ============================================================
    print_section("3. 二维布局 - 混合静态动态");
    
    auto s2xd4 = make_layout(make_shape(Int<2>{}, 4));
    std::cout << "s2xd4 (2x4 混合，列主序):\n";
    print("  Layout: "); print(s2xd4); print("\n");
    print("  2D Layout:\n");
    print_layout(s2xd4);

    // ============================================================
    // 4. 二维布局 - 自定义 stride（带 padding）
    // ============================================================
    print_section("4. 二维布局 - 自定义 stride（带 padding）");
    
    auto s2xd4_a = make_layout(make_shape(Int<2>{}, 4),
                               make_stride(Int<12>{}, Int<1>{}));
    std::cout << "s2xd4_a (2x4，stride=[12,1]，带 padding):\n";
    print("  Layout: "); print(s2xd4_a); print("\n");
    print("  Shape:  "); print(shape(s2xd4_a)); print("\n");
    print("  Stride: "); print(stride(s2xd4_a)); print("\n");
    print("  Size:   "); print(size(s2xd4_a)); print("\n");
    print("  Cosize: "); print(cosize(s2xd4_a)); print(" (注意：实际占用空间)\n");
    print("  2D Layout:\n");
    print_layout(s2xd4_a);

    // ============================================================
    // 5. 二维布局 - 显式列主序
    // ============================================================
    print_section("5. 二维布局 - 显式列主序");
    
    auto s2xd4_col = make_layout(make_shape(Int<2>{}, 4),
                                 LayoutLeft{});
    std::cout << "s2xd4_col (2x4，显式 LayoutLeft):\n";
    print("  Layout: "); print(s2xd4_col); print("\n");
    print("  2D Layout:\n");
    print_layout(s2xd4_col);

    // ============================================================
    // 6. 二维布局 - 行主序
    // ============================================================
    print_section("6. 二维布局 - 行主序");
    
    auto s2xd4_row = make_layout(make_shape(Int<2>{}, 4),
                                 LayoutRight{});
    std::cout << "s2xd4_row (2x4，LayoutRight):\n";
    print("  Layout: "); print(s2xd4_row); print("\n");
    print("  Shape:  "); print(shape(s2xd4_row)); print("\n");
    print("  Stride: "); print(stride(s2xd4_row)); print("\n");
    print("  2D Layout:\n");
    print_layout(s2xd4_row);

    // ============================================================
    // 7. 层次化布局
    // ============================================================
    print_section("7. 层次化布局 - 2x(2x2)");
    
    auto s2xh4 = make_layout(make_shape(2, make_shape(2, 2)),
                             make_stride(4, make_stride(2, 1)));
    std::cout << "s2xh4 (层次化 2x(2x2)):\n";
    print("  Layout: "); print(s2xh4); print("\n");
    print("  Shape:  "); print(shape(s2xh4)); print("\n");
    print("  Stride: "); print(stride(s2xh4)); print("\n");
    print("  Rank:   "); print(rank(s2xh4)); print("\n");
    print("  Depth:  "); print(depth(s2xh4)); print("\n");
    print("  2D Layout:\n");
    print_layout(s2xh4);

    // ============================================================
    // 8. 层次化布局 - 重新解释为列主序
    // ============================================================
    print_section("8. 层次化布局 - 重新解释为列主序");
    
    auto s2xh4_col = make_layout(shape(s2xh4), LayoutLeft{});
    std::cout << "s2xh4_col (使用 s2xh4 的 shape，重新用列主序):\n";
    print("  Layout: "); print(s2xh4_col); print("\n");
    print("  Shape:  "); print(shape(s2xh4_col)); print("\n");
    print("  Stride: "); print(stride(s2xh4_col)); print("\n");
    print("  2D Layout:\n");
    print_layout(s2xh4_col);

    // ============================================================
    // 练习题目
    // ============================================================
    print_section("练习题目");

    std::cout << "\n【题目 1】一维布局 (4):\n";
    auto L1 = make_layout(Int<4>{});
    print("  Layout: "); print(L1); print("\n");
    std::cout << "  一维数组: [";
    for (int i = 0; i < 4; i++) {
        std::cout << L1(i);
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "\n【题目 2】2x3 行主序:\n";
    auto L2 = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          LayoutRight{});
    print("  Layout: "); print(L2); print("\n");
    print_layout(L2);

    std::cout << "\n【题目 3】2x3 列主序:\n";
    auto L3 = make_layout(make_shape(Int<2>{}, Int<3>{}),
                          LayoutLeft{});
    print("  Layout: "); print(L3); print("\n");
    print_layout(L3);

    std::cout << "\n【题目 4】2x4 带 padding:\n";
    auto L4 = make_layout(make_shape(Int<2>{}, Int<4>{}),
                          make_stride(Int<6>{}, Int<1>{}));
    print("  Layout: "); print(L4); print("\n");
    print("  Cosize: "); print(cosize(L4)); print(" (实际占用)\n");
    print_layout(L4);

    std::cout << "\n【题目 5】分层布局 2x(2x3):\n";
    auto L5 = make_layout(make_shape(2, make_shape(2, 3)),
                          make_stride(6, make_stride(3, 1)));
    print("  Layout: "); print(L5); print("\n");
    print("  Shape:  "); print(shape(L5)); print("\n");
    print("  Stride: "); print(stride(L5)); print("\n");
    print_layout(L5);

    // ============================================================
    // 布局兼容性测试
    // ============================================================
    print_section("布局兼容性 (Layout Compatibility)");

    std::cout << "\n测试各种 shape 的兼容性：\n\n";

    // 创建测试 shape
    auto shape_24 = make_shape(24);
    auto shape_32 = make_shape(32);
    auto shape_4x6 = make_shape(Int<4>{}, Int<6>{});
    auto shape_22x6 = make_shape(make_shape(Int<2>{}, Int<2>{}), Int<6>{});
    auto shape_22x32 = make_shape(make_shape(Int<2>{}, Int<2>{}), 
                                   make_shape(Int<3>{}, Int<2>{}));
    auto shape_23x4 = make_shape(make_shape(Int<2>{}, Int<3>{}), Int<4>{});
    auto shape_tuple24 = make_shape(make_tuple(24));

    std::cout << "【示例 1】Shape 24 vs Shape 32\n";
    print("  shape_24:  "); print(shape_24); print("\n");
    print("  shape_32:  "); print(shape_32); print("\n");
    print("  size(24):  "); print(size(shape_24)); print("\n");
    print("  size(32):  "); print(size(shape_32)); print("\n");
    std::cout << "  结果: ❌ 不兼容（size 不相等）\n";

    std::cout << "\n【示例 2】Shape 24 vs Shape (4,6)\n";
    print("  shape_24:    "); print(shape_24); print("\n");
    print("  shape_(4,6): "); print(shape_4x6); print("\n");
    print("  size(24):    "); print(size(shape_24)); print("\n");
    print("  size(4,6):   "); print(size(shape_4x6)); print("\n");
    std::cout << "  结果: ✅ 兼容（24 = 4×6，一维可映射到二维）\n";

    std::cout << "\n【示例 3】Shape (4,6) vs Shape ((2,2),6)\n";
    print("  shape_(4,6):      "); print(shape_4x6); print("\n");
    print("  shape_((2,2),6):  "); print(shape_22x6); print("\n");
    print("  size(4,6):        "); print(size(shape_4x6)); print("\n");
    print("  size((2,2),6):    "); print(size(shape_22x6)); print("\n");
    std::cout << "  结果: ✅ 兼容（4 可以分解为 (2,2)）\n";

    std::cout << "\n【示例 4】Shape ((2,2),6) vs Shape ((2,2),(3,2))\n";
    print("  shape_((2,2),6):       "); print(shape_22x6); print("\n");
    print("  shape_((2,2),(3,2)):   "); print(shape_22x32); print("\n");
    print("  size((2,2),6):         "); print(size(shape_22x6)); print("\n");
    print("  size((2,2),(3,2)):     "); print(size(shape_22x32)); print("\n");
    std::cout << "  结果: ✅ 兼容（6 可以分解为 (3,2)）\n";

    std::cout << "\n【示例 5】Shape 24 vs Shape ((2,2),(3,2)) - 传递性\n";
    print("  shape_24:               "); print(shape_24); print("\n");
    print("  shape_((2,2),(3,2)):    "); print(shape_22x32); print("\n");
    std::cout << "  传递链：24 → (4,6) → ((2,2),6) → ((2,2),(3,2))\n";
    std::cout << "  结果: ✅ 兼容（通过传递性）\n";

    std::cout << "\n【示例 6】Shape ((2,3),4) vs Shape ((2,2),(3,2))\n";
    print("  shape_((2,3),4):        "); print(shape_23x4); print("\n");
    print("  shape_((2,2),(3,2)):    "); print(shape_22x32); print("\n");
    print("  size((2,3),4):          "); print(size(shape_23x4)); print("\n");
    print("  size((2,2),(3,2)):      "); print(size(shape_22x32)); print("\n");
    std::cout << "  分析：\n";
    std::cout << "    ((2,3),4): 外层 6 个元素，内层 4 个元素\n";
    std::cout << "    ((2,2),(3,2)): 外层 4 个元素，内层 6 个元素\n";
    std::cout << "  结果: ❌ 不兼容（层次结构不匹配）\n";

    std::cout << "\n【示例 7】Shape 24 vs Shape (24)\n";
    print("  shape_24:     "); print(shape_24); print("\n");
    print("  shape_(24):   "); print(shape_tuple24); print("\n");
    print("  rank(24):     "); print(rank(shape_24)); print("\n");
    print("  rank((24)):   "); print(rank(shape_tuple24)); print("\n");
    std::cout << "  结果: ✅ 兼容（整数可以变成单元素 tuple）\n";
    std::cout << "        但反过来 (24) → 24 不兼容\n";

    // 实际应用示例
    print_section("兼容性实际应用 - 张量重塑");

    std::cout << "\n将一维数据重塑为多维张量：\n";
    
    // 创建一维布局
    auto layout_1d = make_layout(make_shape(24));
    std::cout << "\n原始一维布局:\n";
    print("  Layout: "); print(layout_1d); print("\n");
    
    // 重塑为 4x6
    auto layout_2d = make_layout(make_shape(Int<4>{}, Int<6>{}));
    std::cout << "\n重塑为 4x6:\n";
    print("  Layout: "); print(layout_2d); print("\n");
    print_layout(layout_2d);
    
    // 重塑为层次化 ((2,2),6)
    auto layout_hier = make_layout(make_shape(make_shape(Int<2>{}, Int<2>{}), Int<6>{}));
    std::cout << "\n重塑为层次化 ((2,2),6):\n";
    print("  Layout: "); print(layout_hier); print("\n");
    print_layout(layout_hier);

    std::cout << "\n这些布局都有 24 个元素，可以相互转换（前提是数据布局匹配）。\n";

    print_section("测试完成！");
    
    return 0;
}

