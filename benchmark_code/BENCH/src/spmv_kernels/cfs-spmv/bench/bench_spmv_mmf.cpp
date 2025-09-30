#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string.h>

#ifdef _MKL
#include <mkl.h>
#include <mkl_spblas.h>
#endif

#ifdef _RSB
#include <blas_sparse.h>
#include <rsb.h>
#endif

#include "cfs.hpp"

extern "C" {
	#include "time_it.h"
}


using namespace std;
using namespace cfs::util;
using namespace cfs::util::memory;
using namespace cfs::util::runtime;
using namespace cfs::matrix::sparse;
using namespace cfs::kernel::sparse;

char *program_name = NULL;

typedef int INDEX;
#ifdef _USE_DOUBLE
typedef double VALUE;
#else
typedef float VALUE;
#endif

static void set_program_name(char *path) {
	if (!program_name)
		program_name = strdup(path);
	if (!program_name)
		fprintf(stderr, "strdup failed\n");
}

static void print_usage() {
	cout << "Usage: " << program_name << " <mmf_file> <format>(0: CSR, 1:SSS, 2: "
		"HYB, 3: MKL-CSR, 4: RSB) <iterations>"
		<< endl;
}

int main(int argc, char **argv) {
	if (argc == 1)
	{
		long buf_n = 1000;
		char buf[buf_n];
		long i;
		i = 0;
		i += snprintf(buf + i, buf_n - i, "%s", "matrix_name");
		i += snprintf(buf + i, buf_n - i, ",%s", "num_threads");
		i += snprintf(buf + i, buf_n - i, ",%s", "csr_m");
		i += snprintf(buf + i, buf_n - i, ",%s", "csr_n");
		i += snprintf(buf + i, buf_n - i, ",%s", "csr_nnz");
		i += snprintf(buf + i, buf_n - i, ",%s", "time");
		i += snprintf(buf + i, buf_n - i, ",%s", "gflops");
		i += snprintf(buf + i, buf_n - i, ",%s", "csr_mem_footprint");
		i += snprintf(buf + i, buf_n - i, ",%s", "W_avg");
		i += snprintf(buf + i, buf_n - i, ",%s", "J_estimated");
		i += snprintf(buf + i, buf_n - i, ",%s", "format_name");
		i += snprintf(buf + i, buf_n - i, ",%s", "m");
		i += snprintf(buf + i, buf_n - i, ",%s", "n");
		i += snprintf(buf + i, buf_n - i, ",%s", "nnz");
		i += snprintf(buf + i, buf_n - i, ",%s", "mem_footprint");
		i += snprintf(buf + i, buf_n - i, ",%s", "mem_ratio");
		i += snprintf(buf + i, buf_n - i, ",%s", "num_loops");
		i += snprintf(buf + i, buf_n - i, ",%s", "preproc_time");
		buf[i] = '\0';
		fprintf(stderr, "%s\n", buf);
		return 0;
	}
	#ifdef _USE_DOUBLE
		printf("_USE_DOUBLE defined, sizeof(VALUE)=%ld\n", sizeof(VALUE));
	#else
		printf("_USE_DOUBLE NOT defined, sizeof(VALUE)=%ld\n", sizeof(VALUE));
	#endif
	set_program_name(argv[0]);
	if (argc < 3) {
		cerr << "Error in number of arguments!" << endl;
		print_usage();
		exit(1);
	}

	const string mmf_file(argv[1]);
	int fmt = atoi(argv[2]);
	if (fmt > 4) {
		cerr << "Error in arguments!" << endl;
		print_usage();
		exit(1);
	}

	int nthreads = get_num_threads();

	// size_t min_num_loops = 128;
	size_t min_num_loops = 256;


	// Load a sparse matrix from an MMF file
	void *mat = nullptr;
	int M = 0, N = 0, nnz = 0;
	string format_string;
	Format format = Format::none;
	switch (fmt) {
		case 0: {
				format = Format::csr;
				format_string = "CSR";
				break;
			}
		case 1: {
				format = Format::sss;
				// format_string = "SSS";
				format_string = "CFS";
				break;
			}
		case 2: {
				format = Format::hyb;
				// format_string = "HYB";
				format_string = "CFH";
				break;
			}
		case 3: {
				format_string = "MKL-CSR";
				break;
			}
		case 4: {
				format_string = "RSB";
				break;
			}
	}

	switch (fmt) {
		case 0:
		case 1:
		case 2: {
				SparseMatrix<INDEX, VALUE> *tmp = SparseMatrix<INDEX, VALUE>::create(mmf_file, format);
				M = tmp->nrows();
				N = tmp->ncols();
				nnz = tmp->nnz();
				mat = (void *)tmp;
				break;
			}
		case 3: {
#ifdef _MKL
				CSRMatrix<INDEX, VALUE> *tmp = new CSRMatrix<INDEX, VALUE>(mmf_file);
				M = tmp->nrows();
				N = tmp->ncols();
				nnz = tmp->nnz();
				mat = (void *)tmp;
#endif
				break;
			}
		case 4:
			break;
	}

	// Prepare vectors
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis_val(0.01, 0.42);

	VALUE *y = (VALUE *)internal_alloc(M * sizeof(VALUE));
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (int i = 0; i < M; i++) {
		y[i] = 0.0;
	}

	VALUE *x = (VALUE *)internal_alloc(N * sizeof(VALUE));
	#pragma omp parallel for schedule(static) num_threads(nthreads)
	for (int i = 0; i < N; i++) {
		x[i] = dis_val(gen);
	}

	double compute_time = 0.0, preproc_time = 0.0, tstart = 0.0, tstop = 0.0,
	       gflops = 0.0;

	if (fmt < 3) {
		SparseMatrix<INDEX, VALUE> *A = (SparseMatrix<INDEX, VALUE> *)mat;

		tstart = omp_get_wtime();
		SpDMV<INDEX, VALUE> spdmv(A);
		tstop = omp_get_wtime();
		preproc_time = tstop - tstart;

#ifdef _LOG_INFO
		cout << "[INFO]: warming up caches..." << endl;
#endif
		// Warm up run
		spdmv(y, M, x, N);
		// for (size_t i = 0; i < min_num_loops / 2; i++)
			// spdmv(y, M, x, N);

#ifdef _LOG_INFO
		cout << "[INFO]: benchmarking SpDMV using " << format_string << "..."
			<< endl;
#endif
		// Benchmark run
		// compute_time = time_it(1,
			// for (size_t i = 0; i < min_num_loops; i++)
				// spdmv(y, M, x, N);
		// );

		volatile unsigned long * L3_cache_block;
		long L3_cache_block_n = atol(getenv("LEVEL3_CACHE_SIZE_TOTAL")) / sizeof(*L3_cache_block);
		L3_cache_block = (typeof(L3_cache_block)) malloc(L3_cache_block_n * sizeof(*L3_cache_block));
		int clear_caches = atoi(getenv("CLEAR_CACHES"));

		compute_time = 0;
		long num_loops = 0;
		while (compute_time < 2.0 || num_loops < min_num_loops)
		{
			if (__builtin_expect(clear_caches, 0))
			{
				if (num_loops >= min_num_loops)
					break;
				_Pragma("omp parallel")
				{
					long i;
					_Pragma("omp for")
					for (i=0;i<L3_cache_block_n;i++)
						L3_cache_block[i] = 0;
				}
			}

			compute_time += time_it(1,
				spdmv(y, M, x, N);
			);
			num_loops++;
		}

		gflops = (((double) num_loops) * 2 * nnz * 1.e-9) / compute_time;
		// cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
			// << " format: " << format_string << " preproc(sec): " << preproc_time
			// << " t(sec): " << compute_time / num_loops << " gflops/s: " << gflops
			// << " threads: " << nthreads
			// << " size(MB): " << A->size() / (float)(1024 * 1024) << endl;
		double csr_mem_footprint = nnz * (sizeof(VALUE) + sizeof(int)) + (M+1) * sizeof(int);
		double mem_footprint = A->size();
		fprintf(stderr, "%s,%d,%d,%d,%d,%lf,%lf,%lf,%d,%d,%s,%d,%d,%d,%lf,%lf,%ld,%lf\n",
				mmf_file.c_str(), nthreads, M, N, nnz,
				compute_time, gflops,
				csr_mem_footprint / (double)(1024 * 1024),
				0, 0,
				format_string.c_str(), M, N, nnz,
				mem_footprint / (double)(1024 * 1024),
				mem_footprint / csr_mem_footprint,
				num_loops,
				preproc_time);

		// Cleanup
		delete A;
	}

	if (fmt == 3) {
#ifdef _MKL
		CSRMatrix<INDEX, VALUE> *A_csr = new CSRMatrix<INDEX, VALUE>(mmf_file);
		VALUE alpha = 1, beta = 0;
		sparse_matrix_t A_view;
		sparse_status_t stat;
		mkl_set_num_threads(nthreads);
#ifdef _USE_DOUBLE
		stat = mkl_sparse_d_create_csr(&A_view, SPARSE_INDEX_BASE_ZERO, M, N,
				A_csr->rowptr(), A_csr->rowptr() + 1,
				A_csr->colind(), A_csr->values());
#else
		stat = mkl_sparse_s_create_csr(&A_view, SPARSE_INDEX_BASE_ZERO, M, N,
				A_csr->rowptr(), A_csr->rowptr() + 1,
				A_csr->colind(), A_csr->values());
#endif
		struct matrix_descr matdescr;
		matdescr.type = SPARSE_MATRIX_TYPE_GENERAL;
		tstart = omp_get_wtime();
		stat = mkl_sparse_set_mv_hint(A_view, SPARSE_OPERATION_NON_TRANSPOSE,
				matdescr, 100000);
		stat = mkl_sparse_set_memory_hint(A_view, SPARSE_MEMORY_AGGRESSIVE);
		stat = mkl_sparse_optimize(A_view);
		tstop = omp_get_wtime();
		preproc_time = tstop - tstart;
		if (stat != SPARSE_STATUS_SUCCESS) {
			cout << "[INFO]: MKL auto-tuning failed" << endl;
		}

#ifdef _LOG_INFO
		cout << "[INFO]: warming up caches..." << endl;
#endif
		// Warm up run
		for (size_t i = 0; i < min_num_loops / 2; i++)
#ifdef _USE_DOUBLE
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
					x, beta, y);
#else
		mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
				x, beta, y);
#endif

#ifdef _LOG_INFO
		cout << "[INFO]: benchmarking SpDMV using MKL-CSR..." << endl;
#endif
		tstart = omp_get_wtime();
		// Benchmark run
		for (size_t i = 0; i < min_num_loops; i++)
#ifdef _USE_DOUBLE
			mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
					x, beta, y);
#else
		mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A_view, matdescr,
				x, beta, y);
#endif
		tstop = omp_get_wtime();
		compute_time = tstop - tstart;

		gflops = ((double)min_num_loops * 2 * nnz * 1.e-9) / compute_time;
		cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
			<< " format: " << format_string << " preproc(sec): " << preproc_time
			<< " t(sec): " << compute_time / min_num_loops << " gflops/s: " << gflops
			<< " threads: " << nthreads << endl;

		mkl_sparse_destroy(A_view);
		delete A_csr;
#endif // _MKL
	}

	if (fmt == 4) {
#ifdef _RSB
		blas_sparse_matrix A = blas_invalid_handle;
		rsb_type_t typecode = RSB_NUMERICAL_TYPE_DOUBLE;
		rsb_err_t errval = RSB_ERR_NO_ERROR;
		VALUE alpha = 1;

		// Initialize library
		if ((errval = rsb_lib_init(RSB_NULL_INIT_OPTIONS)) != RSB_ERR_NO_ERROR) {
			cout << "Error initializing the RSB library" << endl;
			exit(1);
		}

		// Load matrix
		A = rsb_load_spblas_matrix_file_as_matrix_market(mmf_file.c_str(),
				typecode);
		if (A == blas_invalid_handle) {
			cout << "Error while loading matrix from file" << endl;
			exit(1);
		}

		// Autotune
		tstart = omp_get_wtime();
		BLAS_ussp(A, blas_rsb_autotune_next_operation);
		BLAS_dusmv(blas_no_trans, alpha, A, x, 1, y, 1);
		tstop = omp_get_wtime();
		preproc_time = tstop - tstart;

#ifdef _LOG_INFO
		cout << "[INFO]: benchmarking SpDMV using RSB..." << endl;
#endif
		// Benchmark run
		tstart = omp_get_wtime();
		for (size_t i = 0; i < min_num_loops; i++)
			BLAS_dusmv(blas_no_trans, alpha, A, x, 1, y, 1);
		tstop = omp_get_wtime();
		compute_time = tstop - tstart;

		gflops = ((double)min_num_loops * 2 * nnz * 1.e-9) / compute_time;
		cout << setprecision(4) << "matrix: " << basename(mmf_file.c_str())
			<< " format: " << format_string << " preproc(sec): " << preproc_time
			<< " t(sec): " << compute_time / min_num_loops << " gflops/s: " << gflops
			<< " threads: " << nthreads << endl;

		// Cleanup
		BLAS_usds(A);
		if ((errval = rsb_lib_exit(RSB_NULL_EXIT_OPTIONS)) != RSB_ERR_NO_ERROR) {
			cout << "Error finalizing the RSB library" << endl;
			exit(1);
		}

#endif
	}

	// Cleanup
	internal_free(x);
	internal_free(y);
	free(program_name);

	return 0;
}
