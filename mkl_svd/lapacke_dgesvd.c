#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "mkl.h"

#define NUM_WARMUP 2
#define NUM_REPEAT 5

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda ) {
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
    MKL_INT m = 0, n = 0;

    if(argc < 3)
    {
        m = 1024;
        n = 1024;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

	/* Locals */
	MKL_INT lda = m, ldu = m, ldvt = n, info;
	MKL_INT minmn = (m)>(n)?(n):(m);

	/* Local arrays */
	double *s = malloc(n * sizeof(double));
	double *u = malloc(ldu * m * sizeof(double));
	double *vt = malloc(ldvt * n * sizeof(double));
	double *superb = malloc((minmn-1) * sizeof(double));
	double *a_original = malloc(lda * n * sizeof(double));
	double *a = malloc(lda * n * sizeof(double));

	srand((unsigned int)time(NULL));
	random_initialize_matrix(a_original, m, n, lda);

	// print_matrix( "Matrix A", m, n, a, lda );

	/* Executable statements */
	// printf( "LAPACKE_sgesvd (column-major, high-level) Example Program Results\n" );

	memcpy(a, a_original, lda * n * sizeof(double));

	/* Compute SVD */
	info = LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, a, lda,
			s, u, ldu, vt, ldvt, superb );

	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	for(int i = 0; i < NUM_WARMUP; ++i)
	{
		memcpy(a, a_original, lda * n * sizeof(double));

		/* Compute SVD */
		info = LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, a, lda,
				s, u, ldu, vt, ldvt, superb );
	}

    clock_t start, end;
    double cpu_time_used = 0;
	for(int i = 0; i < NUM_REPEAT; ++i)
	{
		memcpy(a, a_original, lda * n * sizeof(double));

		start = clock();
		/* Compute SVD */
		info = LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'A', 'A', m, n, a, lda,
				s, u, ldu, vt, ldvt, superb );
		end = clock();
		cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
	}

	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	printf("LAPACKE SVD (Double) Latency: %lf ms\n", 1000 * cpu_time_used / NUM_REPEAT);
	/* Print singular values */
	// print_matrix( "Singular values", 1, n, s, 1 );
	/* Print left singular vectors */
	// print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
	/* Print right singular vectors */
	// print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
	exit( 0 );
} /* End of LAPACKE_sgesvd Example */

