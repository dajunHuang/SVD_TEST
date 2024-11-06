all: cusolver

cusolver:
	make -C cusolver_svd

clean:
	make -C cusolver_svd clean
