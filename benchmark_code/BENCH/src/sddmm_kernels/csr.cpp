#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "macros/cpp_defines.h"

#include "sddmm_kernel.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "macros/macrolib.h"
	#include "time_it.h"
	#include "parallel_util.h"

	// #define VEC_FORCE

	// #define VEC_X86_512
	// #define VEC_X86_256
	// #define VEC_X86_128
	// #define VEC_ARM_SVE

	#if DOUBLE == 0
		#define VTI   i32
		#define VTF   f32
		#define VTM   m32
		#define VEC_SCALE_SHIFT  2
		// #define VEC_LEN  1
		#define VEC_LEN  vec_len_default_f32
		// #define VEC_LEN  vec_len_default_f64
		// #define VEC_LEN  4
		// #define VEC_LEN  8
		// #define VEC_LEN  16
		// #define VEC_LEN  32
	#elif DOUBLE == 1
		#define VTI   i64
		#define VTF   f64
		#define VTM   m64
		#define VEC_SCALE_SHIFT  3
		#define VEC_LEN  vec_len_default_f64
		// #define VEC_LEN  1
	#endif

	#include "vectorization/vectorization_gen.h"
#ifdef __cplusplus
}
#endif


struct thread_data {
	long i_s;
	long i_e;

	long j_s;
	long j_e;

	// ValueType v_s;
	ValueType v_e;
};


struct CSR : Matrix_Format
{
	INT_T * row_ptr;      // the usual rowptr (of size m+1)
	INT_T * ja;      // the colidx of each NNZ (of size nnz)
	ValueType * a;   // the values (of size NNZ)

	struct thread_data ** tds;

	CSR(INT_T * row_ptr_in, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz) : Matrix_Format(m, n, nnz)
	{
		int num_threads = omp_get_max_threads();
		double time_balance;

		printf("VEC_LEN = %d\n", VEC_LEN);

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

		tds = (typeof(tds)) aligned_alloc(64, num_threads * sizeof(*tds));

		time_balance = time_it(1,
			_Pragma("omp parallel")
			{
				int tnum = omp_get_thread_num();
				struct thread_data * td;
				int use_processes = atoi(getenv("USE_PROCESSES"));
				td = (typeof(td)) aligned_alloc(64, sizeof(*td));
				tds[tnum] = td;
				if (use_processes)
				{
					loop_partitioner_balance_iterations(num_threads, tnum, 0, m, &td->i_s, &td->i_e);
				}
				else
				{
					loop_partitioner_balance_prefix_sums(num_threads, tnum, row_ptr, m, nnz, &td->i_s, &td->i_e);
					// long lower_boundary;
					// loop_partitioner_balance_iterations(num_threads, tnum, 0, nnz, &td->j_s, &td->j_e);
					// macros_binary_search(row_ptr, 0, m, td->j_s, &lower_boundary, NULL);           // Index boundaries are inclusive.
					// td->i_s = lower_boundary;
					// _Pragma("omp barrier")
					// if (tnum == num_threads - 1)   // If we calculate each thread's boundaries individually some empty rows might be unassigned.
						// td->i_e = m;
					// else
						// td->i_e = td->i_s + 1;
				}
			}
		);
		printf("balance time = %g\n", time_balance);
	}

	~CSR()
	{
		free(a);
		free(row_ptr);
		free(ja);
		free(tds);
	}

	void sddmm(long K, ValueType * A, ValueType * B, ValueType * C);
	void statistics_start();
	int statistics_print_data(char * buf, long buf_n);
};


__attribute__((hot,pure))
static inline
ValueType
dot(long K, ValueType * x, ValueType * y)
{
	long i;
	ValueType sum = 0;
	for (i=0;i<K;i++)
	{
		sum += x[i] * y[i];
	}
	return sum;
}

void
compute_csr(CSR * restrict csr, long K, ValueType * A, ValueType * B, ValueType * C)
{
	#pragma omp parallel
	{
		int tnum = omp_get_thread_num();
		struct thread_data * td = csr->tds[tnum];
		long i, i_s, i_e, j, j_s, j_e;
		i_s = td->i_s;
		i_e = td->i_e;
		j_s = td->j_s;
		j_e = td->j_e;

		j_e = csr->row_ptr[i_s];
		for (i=i_s;i<i_e;i++)
		{
			j_s = j_e;
			j_e = csr->row_ptr[i+1];
			if (j_s == j_e)
				continue;
			for (j=j_s;j<j_e;j++)
			{
				C[j] = dot(K, &A[i * K], &B[csr->ja[j] * K]);
			}
		}
	}
}


void
CSR::sddmm(long K, ValueType * A, ValueType * B, ValueType * C)
{
	compute_csr(this, K, A, B, C);
}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz, long symmetric, long symmetry_expanded)
{
	if (symmetric && !symmetry_expanded)
		error("symmetric matrices have to be expanded to be supported by this format");
	struct CSR * csr = new CSR(row_ptr, col_ind, values, m, n, nnz);
	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	csr->format_name = (char *) "Custom_CSR";
	return csr;
}


//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
CSR::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
CSR::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}

