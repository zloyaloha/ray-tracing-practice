NVCC = nvcc
TARGET = main
SRCS = src/main.cu src/camera.cu
INCLUDES = -Iinclude

NVCC_FLAGS = -O3 -std=c++17 -arch=sm_89 --ptxas-options=-v -lineinfo --extra-device-vectorization

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
