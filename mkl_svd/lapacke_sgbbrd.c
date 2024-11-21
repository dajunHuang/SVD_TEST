#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "mkl.h"

#define NUM_WARMUP 1
#define NUM_REPEAT 1

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, MKL_INT m, MKL_INT n, float* a, MKL_INT lda )
{
	MKL_INT i, j;
	printf( "\n %s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
		printf( "\n" );
	}
}

void random_initialize_band_matrix(float* ab, int ku, int kl, size_t n)
{
	int ldab = ku + kl + 1;
	for (int j = 0; j < n; ++j)
	{
		for (int i = 0; i < ku + kl + 1; ++i)
		{
			if(i + j >= ku && i + j <= ku + n)
			{
				ab[i + j * ldab] = rand() / (float) RAND_MAX;
			}
		}
	}
}

/* Main program */
int main(int argc, char *argv[]) {
	mkl_set_num_threads (12);

    MKL_INT m = 1024, n = 1024;
	MKL_INT kl = 32, ku = 32;

    clock_t start, end;
    double brd_cpu_time_used = 0;
	double sqr_cpu_time_used = 0;

    if(argc >= 4)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
		ku = atoi(argv[3]);
		kl = atoi(argv[4]);
    }

	/* Locals */
	MKL_INT ldu = m, ldq = m, ldpt = n, info;
	MKL_INT ldab = kl + ku + 1;
	MKL_INT minmn = (m)>(n)?(n):(m);

	/* Local arrays */
	float *ab_original = malloc(ldab * n * sizeof(float));
	float *ab = malloc(ldab * n * sizeof(float));
	float *d_original = malloc(minmn * sizeof(float));
	float *e_original = malloc((minmn - 1) * sizeof(float));
	float *d = malloc(minmn * sizeof(float));
	float *e = malloc((minmn - 1) * sizeof(float));
	float *q = malloc(ldq * m * sizeof(float));
	float *pt = malloc(ldq * n * sizeof(float));


	srand((unsigned int)time(NULL));
	random_initialize_band_matrix(ab_original, ku, kl, n);

	memcpy(ab, ab_original, ldab * n * sizeof(float));

	// print_matrix( "Matrix ab", kl + ku + 1, n, ab, ldab );

	info = LAPACKE_sgbbrd(LAPACK_COL_MAJOR, 'N', m, n, 0, kl, ku, ab, ldab, d_original, e_original, q, ldq, pt, ldpt, NULL, 1);

	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	for(int i = 0; i < NUM_WARMUP; ++i)
	{
		memcpy(ab, ab_original, ldab * n * sizeof(float));
		info = LAPACKE_sgbbrd(LAPACK_COL_MAJOR, 'N', m, n, 0, kl, ku, ab, ldab, d_original, e_original, q, ldq, pt, ldpt, NULL, 1);
	}

	for(int i = 0; i < NUM_REPEAT; ++i)
	{
		memcpy(ab, ab_original, ldab * n * sizeof(float));
		start = clock();
		info = LAPACKE_sgbbrd(LAPACK_COL_MAJOR, 'N', m, n, 0, kl, ku, ab, ldab, d_original, e_original, q, ldq, pt, ldpt, NULL, 1);
		end = clock();
		brd_cpu_time_used += ((float) (end - start)) / CLOCKS_PER_SEC;
	}

	/* Check for convergence */
	if( info > 0 ) {
		printf( "The algorithm computing SVD failed to converge.\n" );
		exit( 1 );
	}

	printf("LAPACKE GBBRD (Float) Latency: %lf ms\n", 1000 * brd_cpu_time_used / NUM_REPEAT);
	// print_matrix( "Diagonal values", 1, minmn, d, 1 );
	// print_matrix( "Off-Diagonal values", 1, minmn - 1, e, 1 );

	// memcpy(d, d_original, minmn * sizeof(float));
	// memcpy(e, e_original, (minmn - 1) * sizeof(float));
	// // info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', minmn, 0, 0, 0, 
	// // 	d, e, NULL, 1, NULL, 1, NULL, 1); 
	// info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', minmn, d, e, 
	// 	NULL, 1, NULL, 1, NULL, NULL);

	// if( info > 0 ) {
	// 	printf( "The algorithm computing SVD failed to converge.\n" );
	// 	exit( 1 );
	// }

	// for(int i = 0; i < NUM_WARMUP; ++i)
	// {
	// 	memcpy(d, d_original, minmn * sizeof(float));
	// 	memcpy(e, e_original, (minmn - 1) * sizeof(float));
	// 	// info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', minmn, 0, 0, 0, 
	// 	// 	d, e, NULL, 1, NULL, 1, NULL, 1); 
	// 	info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', minmn, d, e, 
	// 		NULL, 1, NULL, 1, NULL, NULL);
	// }

	// for(int i = 0; i < NUM_REPEAT; ++i)
	// {
	// 	start = clock();
	// 	memcpy(d, d_original, minmn * sizeof(float));
	// 	memcpy(e, e_original, (minmn - 1) * sizeof(float));
	// 	// info = LAPACKE_dbdsqr(LAPACK_COL_MAJOR, 'U', minmn, 0, 0, 0, 
	// 	// 	d, e, NULL, 1, NULL, 1, NULL, 1); 
	// 	info = LAPACKE_dbdsdc(LAPACK_COL_MAJOR, 'U', 'N', minmn, d, e, 
	// 		NULL, 1, NULL, 1, NULL, NULL);
	// 	end = clock();
	// 	sqr_cpu_time_used += ((float) (end - start)) / CLOCKS_PER_SEC;
	// }

	// if( info > 0 ) {
	// 	printf( "The algorithm computing SVD failed to converge.\n" );
	// 	exit( 1 );
	// }

	// printf("LAPACKE GBBRD SQR (float) Latency: %lf ms\n", 1000 * sqr_cpu_time_used / NUM_REPEAT);

	exit( 0 );
} /* End of LAPACKE_sgesvd Example */

