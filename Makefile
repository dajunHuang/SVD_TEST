all: cusolver mkl

cusolver:
	make -C cusolver_svd

mkl:
	make -C mkl_svd

clean:
	make -C cusolver_svd clean
	make -C mkl_svd clean
