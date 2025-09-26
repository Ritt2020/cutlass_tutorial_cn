# Makefile for CUTLASS GEMM test
# 编译 CUTLASS 半精度 GEMM 示例
# 请修改为自己的 CUTLASS 路径、CUDA 路径、编译器路径、架构等。

# 编译器设置
NVCC = nvcc
CXX = g++

# CUTLASS 路径设置
CUTLASS_ROOT = /sharedir44-48/wanghy/cutlass
CUTLASS_INCLUDE = $(CUTLASS_ROOT)/include
CUTLASS_UTIL_INCLUDE = $(CUTLASS_ROOT)/tools/util/include

# 包含路径
INCLUDES = -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL_INCLUDE)

# 编译选项
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_80 -w --expt-relaxed-constexpr
CXX_FLAGS = -O3 -std=c++17 -w

# 目标文件
TARGET = test_gemm
SOURCE = test.cu

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): $(SOURCE)
	$(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)

# 清理目标
clean:
	rm -f $(TARGET) *.o

# 运行目标
run: $(TARGET)
	./$(TARGET)

# 运行带参数的目标
run-args: $(TARGET)
	./$(TARGET) 256 256 256 1.0 0.0

# 检查设备信息
check-device:
	@echo "检查 CUDA 设备信息："
	nvidia-smi

# 帮助信息
help:
	@echo "可用的 make 目标："
	@echo "  all        - 编译程序 (默认)"
	@echo "  clean      - 清理生成的文件"
	@echo "  run        - 运行程序 (使用默认参数)"
	@echo "  run-args   - 运行程序 (使用自定义参数 256x256x256)"
	@echo "  check-device - 检查 CUDA 设备信息"
	@echo "  help       - 显示此帮助信息"
	@echo ""
	@echo "编译选项："
	@echo "  NVCC_FLAGS = $(NVCC_FLAGS)"
	@echo "  INCLUDES   = $(INCLUDES)"

# 声明伪目标
.PHONY: all clean run run-args check-device help