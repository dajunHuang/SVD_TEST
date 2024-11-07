C_FILES = $(shell find ./ -name '*.c')

all: $(C_FILES)
	/opt/intel/oneapi/compiler/2024.2/bin/icx  -qmkl -isystem /opt/intel/oneapi/mkl/2024.2/include -O3 -DNDEBUG -DMKL_ILP64 -std=gnu99 -o lapacke_dgesvd lapacke_dgesvd.c
	objdump -d lapacke_dgesvd > lapacke_dgesvd.asm

clean:
	-rm lapacke_dgesvd
