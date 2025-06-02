#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "macros/cpp_defines.h"

#include "spmv_bench_common.h"
#include "spmv_kernel.h"
#include <riscv_vector.h>

// #ifdef RAVE_TRACING
// 	#include <sdv_tracing.h>
// #endif

#ifdef __cplusplus
extern "C"{
#endif
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "parallel_util.h"
	#include "array_metrics.h"

	#include "radix_sort.h"
	#include "sellcs-spmv.h"
	#include "sellcs_utils.h"
	#include <string.h>

#ifdef __cplusplus
}
#endif

extern int prefetch_distance;

struct CSRArrays : Matrix_Format
{
	INT_T * row_ptr;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)

	ValueType * a;   // the values (of size NNZ)

	long num_loops;

	sellcs_matrix_t * sellcs_mtx; // sell-c-sigma sellcs_mtx structure

	index_t * row_ptr64;      // the usual rowptr (of size m+1)
	index_t * ja64;      // the colidx of each NNZ (of size nnz)

	CSRArrays(INT_T * row_ptr_in, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		int num_threads = omp_get_max_threads();
		double time_balance;

		row_ptr = (typeof(row_ptr)) aligned_alloc(64, (m+1) * sizeof(*row_ptr));
		ja = (typeof(ja)) aligned_alloc(64, nnz * sizeof(*ja));
		a = (typeof(a)) aligned_alloc(64, nnz * sizeof(*a));

		#pragma omp parallel for
		for (long i=0;i<m+1;i++)
			row_ptr[i] = row_ptr_in[i];
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
		{
			a[i]=values[i];
			ja[i]=col_ind[i];
		}

		/*************************************************************************************/
		sellcs_mtx = (sellcs_matrix_t *) malloc(sizeof(sellcs_matrix_t));
		
		const uint64_t max_vlen = 256;
		const uint64_t sigma_window = 16384;
		sellcs_init_params(max_vlen, sigma_window, sellcs_mtx);

		// Convert CSR to sell-c-sigma
		// These (row_ptr64, ja64) is needed because we compile the sell-c-s project with the INDEX64 flag still =1 ... But this needs to change in the future.
		row_ptr64 = (index_t *) aligned_alloc(64, (m+1) * sizeof(index_t));
		ja64 = (index_t *) aligned_alloc(64, nnz * sizeof(index_t));
		#pragma omp parallel for
		for (long i=0;i<m+1;i++)
			row_ptr64[i] = row_ptr[i];
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
			ja64[i]=ja[i];
		sellcs_create_matrix_from_CSR_rd(m, n, row_ptr64, ja64, a, 0, 0, sellcs_mtx);

		// Auto-tune execution parameters
		sellcs_analyze_matrix(sellcs_mtx, 0);
		/*************************************************************************************/
	}

	~CSRArrays()
	{
		free(a);
		free(row_ptr);
		free(ja);

		free(sellcs_mtx);
	}

	void spmv(ValueType * x, ValueType * y);
	void statistics_start();
	int statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n);
};


void compute_csr(CSRArrays * restrict csr, ValueType * restrict x , ValueType * restrict y);


void
CSRArrays::spmv(ValueType * x, ValueType * y)
{
	num_loops++;
	compute_csr(this, x, y);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz, long symmetric, long symmetry_expanded)
{
	if (symmetric && !symmetry_expanded)
		error("symmetric matrices have to be expanded to be supported by this format");
	struct CSRArrays * csr = new CSRArrays(row_ptr, col_ind, values, m, n, nnz);
	// for (long i=0;i<10;i++)
		// printf("%d\n", row_ptr[i]);
	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	csr->format_name = (char *) "SELL_C_s";
	return csr;
}


//==========================================================================================================================================
//= Subkernels CSR
//==========================================================================================================================================

//==========================================================================================================================================
//= CSR Custom
//==========================================================================================================================================


void
compute_csr(CSRArrays * restrict csr, ValueType * restrict x, ValueType * restrict y)
{
	// Tracing needs to happen inside the function that is called (in the sellcs_mv_kernels_epi.c file), so remove it from here.

	// #ifdef RAVE_TRACING
	// 	trace_event_and_value(1000, 1);
	// #endif
	
	sellcs_execute_mv_d(csr->sellcs_mtx, x, y);

	// #ifdef RAVE_TRACING
	// 	trace_event_and_value(1000, 0);
	// #endif
}

//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
CSRArrays::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
CSRArrays::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}

