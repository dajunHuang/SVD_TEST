C_FILES = $(shell find ./ -name '*.c')

OBJ_FILES = $(C_FILES:%.c=%)

all: $(OBJ_FILES)

%: %.c
	nvcc -arch=native -g -l magma_sparse -l cusolver -l cublas -I . -I /home/djhuang/magma_build/include -O3 -DDEBUG $< -o $@

clean:
	-rm $(OBJ_FILES)
