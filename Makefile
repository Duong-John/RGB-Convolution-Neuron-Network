NVCC = nvcc

ARCH = -arch=sm_100 

CUDA_DIR = $(subst \,/,$(CUDA_PATH))

CXXFLAGS = -w -ISDL/include -Iinclude -Itensor -I"$(CUDA_DIR)/include" -std=c++20


LDFLAGS  = -LSDL/lib1 -LSDL/lib2 -lSDL2 -lSDL2main -lSDL2_image -lcudart -lcuda -lcublas

# FILES & DIRECTORIES
BUILD_DIR = build
SRCS = src/Dataset.cpp src/Convolution.cpp src/ReLU.cpp src/Max_Pooling.cpp \
       src/Utils.cpp src/Linear.cpp src/Softmax.cpp src/Loss.cpp \
       src/Optimizer.cpp src/Layer.cpp src/Visual.cpp src/Model.cpp src/Dropout.cpp src/Program.cpp \
       src/Drive_Singleton.cpp

OBJS = $(patsubst src/%.cpp, $(BUILD_DIR)/%.obj, $(SRCS))
PTX_FILE = kernel/conv_kernel.ptx
CU_SRC   = kernel/kernel.cu

all: program $(PTX_FILE)

# .cpp -> .obj NVCC
$(BUILD_DIR)/%.obj: src/%.cpp
	@echo "=> Compiling C++ with NVCC: $<"
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -c $< -o $@ $(CXXFLAGS)

$(PTX_FILE): $(CU_SRC)
	@echo "=> Compiling CUDA to PTX: $<"
	$(NVCC) $(ARCH) -ptx $< -o $@

# Link 
program: $(OBJS)
	@echo "=> Linking with NVCC..."
	$(NVCC) $(OBJS) -o program $(LDFLAGS)
	@echo "=> Complete Compiling and Linking!"

clean:
	@echo "${BLUE}=> Cleaning project...${NC}"
	rm -rf $(BUILD_DIR)
	rm -f program.exe $(PTX_FILE)
	rm -f program.exp program.lib
	@echo "${GREEN}=> Complete cleaning!${NC}"