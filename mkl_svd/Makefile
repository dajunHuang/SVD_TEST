C_FILES = $(shell find ./ -name '*.c')

OBJ_FILES = $(C_FILES:%.c=%)

all: $(OBJ_FILES)

%: %.c
	/opt/intel/oneapi/compiler/2024.2/bin/icx -g -lmkl_rt -qmkl -isystem /opt/intel/oneapi/mkl/2024.2/include -O3 -DDEBUG  $< -o $@

clean:
	-rm $(OBJ_FILES)
