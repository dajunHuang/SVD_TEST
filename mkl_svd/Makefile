C_FILES = $(shell find ./ -name '*.c')

OBJ_FILES = $(C_FILES:%.c=%)

%: %.c
	/opt/intel/oneapi/compiler/2024.2/bin/icx  -qmkl -isystem /opt/intel/oneapi/mkl/2024.2/include -O3 -DNDEBUG -DMKL_ILP64 -std=gnu99 $< -o $@

all: $(OBJ_FILES)

clean:
	-rm $(OBJ_FILES)
