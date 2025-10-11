# Makefile for CuTe Tensor Example

# CUTLASS 路径
CUTLASS_DIR = ../../cutlass

# 编译器
NVCC = nvcc

# 编译选项
NVCC_FLAGS = -std=c++17 \
             -I$(CUTLASS_DIR)/include \
             -arch=sm_90a \
             -O3 \
             --expt-relaxed-constexpr

# 目标文件
TARGET = cute_tensor

# 源文件
SOURCES = cute_tensor.cu

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET)

# 帮助信息
help:
	@echo "可用的 make 目标："
	@echo "  make         - 编译程序"
	@echo "  make run     - 编译并运行程序"
	@echo "  make clean   - 清理生成的文件"
	@echo "  make help    - 显示此帮助信息"

.PHONY: all run clean help

