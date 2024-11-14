#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "mkl.h"

#define NUM_WARMUP 1
#define NUM_REPEAT 1

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda )
{
	MKL_INT i, j;
	printf( "\n %s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
		printf( "\n" );
	}
}

void random_initialize_matrix(double* A, size_t m, size_t n, size_t lda)
{
    for (size_t j = 0; j < n; ++j)
    {
        for (size_t i = 0; i < m; ++i)
        {
            A[i + j * lda] = rand() / (double) RAND_MAX;
        }
    }
}

/* Main program */
int main(int argc, char *argv[]) {
    MKL_INT m = 1024, n = 1024;

    clock_t start, end;
	double sqr_cpu_time_used = 0;

    if(argc >= 3)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

	/* Locals */
	MKL_INT lda = m, info;
	MKL_INT minmn = (m)>(n)?(n):(m);

	/* Local arrays */
	double *d_original = malloc(minmn * sizeof(double));
	double *e_original = malloc((minmn - 1) * sizeof(double));
	double *d = malloc(minmn * sizeof(double));
	double *e = malloc((minmn - 1) * sizeof(double));

	srand((unsigned int)time(NULL));
	random_initialize_matrix(d_original, 1, minmn, 1);
	random_initialize_matrix(e_original, 1, minmn - 1, 1);

	for(int i = 0; i < NUM_WARMUP; ++i)
	{
		memcpy(d, d_original, minmn * sizeof(double));
		memcpy(e, e_original, (minmn - 1) * sizeof(double));
		info = LAPACKE_dsterf(minmn, d, e);
	}

	for(int i = 0; i < NUM_REPEAT; ++i)
	{
		memcpy(d, d_original, minmn * sizeof(double));
		memcpy(e, e_original, (minmn - 1) * sizeof(double));
		start = clock();
		info = LAPACKE_dsterf(minmn, d, e);
		end = clock();
		sqr_cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
	}

	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	printf("LAPACKE symmetric tridiagonal to diagonal (Double) Latency: %lf ms\n", 1000 * sqr_cpu_time_used / NUM_REPEAT);

	// print_matrix( "Diagonal values", 1, minmn, d, 1 );

	exit( 0 );
} /* End of LAPACKE_sgesvd Example */

