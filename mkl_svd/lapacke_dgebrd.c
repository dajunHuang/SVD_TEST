#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "mkl.h"

#define NUM_WARMUP 2
#define NUM_REPEAT 5

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
    double brd_cpu_time_used = 0;
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
	double *a_original = malloc(lda * n * sizeof(double));
	double *a = malloc(lda * n * sizeof(double));
	double *d_original = malloc(minmn * sizeof(double));
	double *e_original = malloc((minmn - 1) * sizeof(double));
	double *d = malloc(minmn * sizeof(double));
	double *e = malloc((minmn - 1) * sizeof(double));
	double *tauq = malloc(minmn * sizeof(double));
	double *taup = malloc(minmn * sizeof(double));

	srand((unsigned int)time(NULL));
	random_initialize_matrix(a_original, m, n, lda);

	memcpy(a, a_original, lda * n * sizeof(double));

	// print_matrix( "Matrix a", m, n, a, lda);

	info = LAPACKE_dgebrd(LAPACK_COL_MAJOR, m, n, a, lda, d_original, 
		e_original, tauq, taup);

	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	for(int i = 0; i < NUM_WARMUP; ++i)
	{
		memcpy(a, a_original, lda * n * sizeof(double));
		info = LAPACKE_dgebrd(LAPACK_COL_MAJOR, m, n, a, lda, 
			d_original, e_original, tauq, taup);
	}

	for(int i = 0; i < NUM_REPEAT; ++i)
	{
		memcpy(a, a_original, lda * n * sizeof(double));
		start = clock();
		info = LAPACKE_dgebrd(LAPACK_COL_MAJOR, m, n, a, lda, 
			d_original, e_original, tauq, taup);
		end = clock();
		brd_cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
	}

	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	printf("LAPACKE GEBRD (Double) Latency: %lf ms\n", 1000 * brd_cpu_time_used / NUM_REPEAT);

	// print_matrix( "Diagonal values", 1, minmn, d_original, 1 );
	// print_matrix( "Off-Diagonal values", 1, minmn - 1, e_original, 1 );

	memcpy(d, d_original, minmn * sizeof(double));
	memcpy(e, e_original, (minmn - 1) * sizeof(double));
	// info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', minmn, 0, 0, 0, 
	// 	d, e, NULL, 1, NULL, 1, NULL, 1); 
	info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', minmn, d, e, 
		NULL, 1, NULL, 1, NULL, NULL);

	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	for(int i = 0; i < NUM_WARMUP; ++i)
	{
		memcpy(d, d_original, minmn * sizeof(double));
		memcpy(e, e_original, (minmn - 1) * sizeof(double));
		// info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', minmn, 0, 0, 0, 
		// 	d, e, NULL, 1, NULL, 1, NULL, 1); 
		info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', minmn, d, e, 
			NULL, 1, NULL, 1, NULL, NULL);
	}

	for(int i = 0; i < NUM_REPEAT; ++i)
	{
		start = clock();
		memcpy(d, d_original, minmn * sizeof(double));
		memcpy(e, e_original, (minmn - 1) * sizeof(double));
		// info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', minmn, 0, 0, 0, 
		// 	d, e, NULL, 1, NULL, 1, NULL, 1); 
		info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', minmn, d, e, 
			NULL, 1, NULL, 1, NULL, NULL);
		end = clock();
		sqr_cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
	}

	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	printf("LAPACKE GEBRD DC (Double) Latency: %lf ms\n", 1000 * sqr_cpu_time_used / NUM_REPEAT);

	// print_matrix( "Diagonal values", 1, minmn, d, 1 );

	exit( 0 );
} /* End of LAPACKE_sgesvd Example */

