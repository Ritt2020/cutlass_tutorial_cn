# Makefile for CuTe Copy Examples

# CUTLASS 路径
CUTLASS_DIR = ../../cutlass

# 编译器
NVCC = nvcc

# 基础编译选项
NVCC_FLAGS_BASE = -std=c++17 \
                   -I$(CUTLASS_DIR)/include \
                   -O3 \
                   --expt-relaxed-constexpr

# SM90 编译选项（用于基本示例）
NVCC_FLAGS_SM90 = $(NVCC_FLAGS_BASE) -arch=sm_90a

# SM80 编译选项（用于 Copy_Atom 示例）
NVCC_FLAGS_SM80 = $(NVCC_FLAGS_BASE) -arch=sm_80

# 目标文件
TARGETS = 01_basic_copy \
          02_copy_if \
          03_copy_atom

# 默认目标
all: $(TARGETS)

# 基本拷贝示例（适用于所有架构）
01_basic_copy: 01_basic_copy.cu
	$(NVCC) $(NVCC_FLAGS_SM90) -o $@ $<

# Copy_if 示例（适用于所有架构）
02_copy_if: 02_copy_if.cu
	$(NVCC) $(NVCC_FLAGS_SM90) -o $@ $<

# Copy_Atom 示例（需要 SM80+）
03_copy_atom: 03_copy_atom.cu
	$(NVCC) $(NVCC_FLAGS_SM80) -o $@ $<

# Copy zfill 示例（需要 SM80+）
04_copy_zfill: 04_copy_zfill.cu
	$(NVCC) $(NVCC_FLAGS_SM80) -o $@ $<

# 运行所有示例
run: $(TARGETS)
	@echo "=== 运行 01_basic_copy ==="
	./01_basic_copy
	@echo ""
	@echo "=== 运行 02_copy_if ==="
	./02_copy_if
	@echo ""
	@echo "=== 运行 03_copy_atom (需要 SM80+) ==="
	./03_copy_atom || echo "跳过（需要 Ampere 架构或更高）"

# 只运行基本示例（不需要 SM80）
run-basic: 01_basic_copy 02_copy_if
	@echo "=== 运行 01_basic_copy ==="
	./01_basic_copy
	@echo ""
	@echo "=== 运行 02_copy_if ==="
	./02_copy_if

# 只运行 SM80 示例
run-sm80: 03_copy_atom 04_copy_zfill
	@echo "=== 运行 03_copy_atom ==="
	./03_copy_atom
	@echo ""
	@echo "=== 运行 04_copy_zfill ==="
	./04_copy_zfill

# 清理
clean:
	rm -f $(TARGETS)

# 帮助信息
help:
	@echo "可用的 make 目标："
	@echo "  make              - 编译所有程序"
	@echo "  make 01_basic_copy    - 编译基本拷贝示例"
	@echo "  make 02_copy_if       - 编译条件拷贝示例"
	@echo "  make 03_copy_atom     - 编译 Copy_Atom 示例（需要 SM80+）"
	@echo "  make 04_copy_zfill    - 编译 zfill 示例（需要 SM80+）"
	@echo "  make run          - 编译并运行所有程序"
	@echo "  make run-basic    - 运行基本示例"
	@echo "  make run-sm80     - 运行 SM80 示例"
	@echo "  make clean        - 清理生成的文件"
	@echo "  make help         - 显示此帮助信息"
	@echo ""
	@echo "注意："
	@echo "  - 03_copy_atom 和 04_copy_zfill 需要 Ampere (SM80) 或更高架构"
	@echo "  - 如果你的 GPU 是 Hopper (SM90)，可以使用 -arch=sm_90a"
	@echo "  - 如果你的 GPU 是 Ampere (SM80)，可以使用 -arch=sm_80"

.PHONY: all run run-basic run-sm80 clean help

