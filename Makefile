CXX = g++
NVCC = nvcc

ESC := $(shell printf '\033')

RED := $(ESC)[0;31m
GREEN := $(ESC)[0;32m
BLUE := $(ESC)[0;34m
NC := $(ESC)[0m

ifneq ($(OS),Windows_NT)
$(error $(RED)Error: This project currently only supports Windows OS.$(NC))
else
$(info $(GREEN)=> This is Window Environment -> Compatible with the program.$(NC))
endif

ifeq ($(CUDA_PATH),)
$(error $(RED)Error: CUDA_PATH environment variable is not found! Please install NVIDIA CUDA Toolkit or set the variable manually.$(NC))
else
$(info $(GREEN)=> Found CUDA_PATH environment variable: $(CUDA_PATH) $(NC))
$(info $(GREEN)=> Compiling process starts.$(NC))
endif

# PATHS & FLAGS
# CUDA_PATH ?= C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.9
CUDA_DIR = $(subst \,/,$(CUDA_PATH))

CXXFLAGS = -w -ISDL/include -Iinclude -Itensor -I"$(CUDA_DIR)/include" -std=c++20
LDFLAGS  = -LSDL/lib1 -LSDL/lib2 -L"$(CUDA_DIR)/lib/x64" -lSDL2 -lSDL2_image -lcudart -lcuda

NVCCFLAGS = -w -arch=sm_89 -ptx

# FILES & DIRECTORIES
BUILD_DIR = build

SRCS = src/Dataset.cpp src/Convolution.cpp src/ReLU.cpp src/Max_Pooling.cpp \
       src/Utils.cpp src/Linear.cpp src/Softmax.cpp src/Loss.cpp \
       src/Optimizer.cpp src/Layer.cpp src/Model.cpp src/Dropout.cpp src/Program.cpp \
       src/Drive_Singleton.cpp

OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# File CUDA
PTX_FILE = kernel/conv_kernel.ptx
CU_SRC   = kernel/kernel.cu


# BUILD RULES
all: program $(PTX_FILE)

# CUDA (.cu) -> (.ptx)
$(PTX_FILE): $(CU_SRC)
	@echo "${BLUE}=> Compiling file CUDA:${NC} $<"
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@echo "${BLUE}=> Compiling C++:${NC} $<"
	@mkdir -p $(BUILD_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

# Link file .o 
program: $(OBJS)
	@echo "${BLUE}=> Linking...${NC}"
	$(CXX) $(OBJS) -o program $(LDFLAGS)
	@echo "${GREEN}=> Complete Compiling and Linking!${NC}"
	@echo "${BLUE}=> Run program by typing ./program${NC}"

# UTILITIES
clean:
	@echo "=>${BLUE}=> Cleaning project...${NC}"
	rm -rf $(BUILD_DIR) program $(PTX_FILE)
	@echo "${GREEN}=> Complete cleaning!${NC}"