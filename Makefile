all: cusolver mkl ozimmu tcgemm

cusolver:
	make -C cusolver_svd

mkl:
	make -C mkl_svd

ozimmu:
	make -C ozimmu

tcgemm:
	make -C tcgemm

clean:
	make -C cusolver_svd clean
	make -C mkl_svd clean
	make -C ozimmu clean
	make -C tcgemm clean
	make -C PanelQR_ori clean
	make -C PanelQR1 clean
