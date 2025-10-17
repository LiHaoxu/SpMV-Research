#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>

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


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))

#define ValueType double
#define ValueTypeReference double

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)
#define THRESHOLD (16*1)
#define BH (128*1)
#define LOG_BH (7)
#define BW (128*1)
#define MIN_OCC (BW*3/4)
//#define MIN_OCC (-1)
#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define SC_SIZE (2048)


struct thread_data {
	long i_s;
	long i_e;

	long j_s;
	long j_e;

	// ValueType v_s;
	ValueType v_e;
};


struct ASpT : Matrix_Format
{
	INT_T * row_ptr;
	INT_T * ja;
	ValueType * a;

	long npanel;
	double vari;
	INT_T * mcsr_e;
	INT_T * mcsr_cnt;
	INT_T * special;
	INT_T * special2;
	long special_p;

	ASpT(INT_T * row_ptr_in, INT_T * col_ind, ValueTypeReference * values, long m_orig, long n, long nnz) : Matrix_Format(m_orig, n, nnz)
	{
		long num_threads = omp_get_max_threads();
		INT_T *mcsr_chk;
		double avg0[num_threads];
		char scr_pad[num_threads][SC_SIZE];

		m = CEIL(m_orig,BH)*BH;
		// printf("m=%ld m_orig=%ld\n", m, m_orig);


		// This format uses the matrix values, so change them to 1 for the validation.
		#pragma omp parallel for
		for(long i=0;i<nnz;i++)
			values[i] = 1.0;


		row_ptr = (INT_T *)aligned_alloc(64, sizeof(INT_T)*(m+1));
		for (long i=0;i<m_orig+1;i++)
			row_ptr[i] = row_ptr_in[i];
		for (long i=m_orig+1;i<m+1;i++)
			row_ptr[i] = row_ptr[m_orig];

		npanel = CEIL(m,BH);

		ja = (INT_T *)aligned_alloc(64, sizeof(INT_T)*nnz+256);
		a = (ValueType *)aligned_alloc(64, sizeof(ValueType)*nnz+256);

		special = (INT_T *)malloc(sizeof(INT_T)*nnz);
		special2 = (INT_T *)malloc(sizeof(INT_T)*nnz);
		memset(special, 0, sizeof(INT_T)*nnz);
		memset(special2, 0, sizeof(INT_T)*nnz);


		mcsr_cnt = (INT_T *)malloc(sizeof(INT_T)*(npanel+1));
		mcsr_chk = (INT_T *)malloc(sizeof(INT_T)*(npanel+1));
		mcsr_e = (INT_T *)malloc(sizeof(INT_T)*nnz); // reduced later
		memset(mcsr_cnt, 0, sizeof(INT_T)*(npanel+1));
		memset(mcsr_chk, 0, sizeof(INT_T)*(npanel+1));
		memset(mcsr_e, 0, sizeof(INT_T)*nnz);	

		long bv_size = CEIL(n, 32);
		UINT_T ** bv = (UINT_T **)malloc(sizeof(UINT_T *)*num_threads);
		for(long i=0;i<num_threads;i++) 
			bv[i] = (UINT_T *)malloc(sizeof(UINT_T)*bv_size);
		INT_T **csr_e1 = (INT_T **)malloc(sizeof(INT_T *)*2);
		short **coo = (short **)malloc(sizeof(short *)*2);
		for(long i=0;i<2;i++) {
			csr_e1[i] = (INT_T *)malloc(sizeof(INT_T)*nnz);
			coo[i] = (short *)malloc(sizeof(short)*nnz);
		}

		// filtering(WILL)
		//memcpy(csr_e1[0], col_ind, sizeof(INT_T)*nnz);
		#pragma omp parallel for schedule(dynamic, 1)
		for(long row_panel=0; row_panel<m/BH; row_panel++) {
			for(long i=row_panel*BH; i<(row_panel+1)*BH; i++) {
				for(long j=row_ptr[i]; j<row_ptr[i+1]; j++) {
					csr_e1[0][j] = col_ind[j];
				}
			}

		}

		#pragma omp parallel for schedule(dynamic, 1)
		for(long row_panel=0; row_panel<m/BH; row_panel++) {
			long tid = omp_get_thread_num();
			long i, j, t_sum=0;

			// coo generate and mcsr_chk
			memset(scr_pad[tid], 0, sizeof(char)*SC_SIZE);
			for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
				for(j=row_ptr[i]; j<row_ptr[i+1]; j++) {
					coo[0][j] = (i&(BH-1));
					long k = (col_ind[j]&(SC_SIZE-1));
					if(scr_pad[tid][k] < THRESHOLD) {
						if(scr_pad[tid][k] == THRESHOLD - 1) t_sum++;
						scr_pad[tid][k]++;
					}
				}
			}

			if (t_sum < MIN_OCC) {
				mcsr_chk[row_panel] = 1;
				mcsr_cnt[row_panel+1] = 1;
				continue;
			} 

			// sorting(merge sort)
			long flag = 0;
			for(long stride = 1; stride <= BH/2; stride *= 2, flag=1-flag) {
				for(long pivot = row_panel*BH; pivot < (row_panel+1)*BH; pivot += stride*2) {
					long l1, l2;
					for(i = l1 = row_ptr[pivot], l2 = row_ptr[pivot+stride]; l1 < row_ptr[pivot+stride] && l2 < row_ptr[pivot+stride*2]; i++) {
						if(csr_e1[flag][l1] <= csr_e1[flag][l2]) {
							coo[1-flag][i] = coo[flag][l1];
							csr_e1[1-flag][i] = csr_e1[flag][l1++];
						}
						else {
							coo[1-flag][i] = coo[flag][l2];
							csr_e1[1-flag][i] = csr_e1[flag][l2++];	
						}
					}
					while(l1 < row_ptr[pivot+stride]) {
						coo[1-flag][i] = coo[flag][l1];
						csr_e1[1-flag][i++] = csr_e1[flag][l1++];
					}
					while(l2 < row_ptr[pivot+stride*2]) {
						coo[1-flag][i] = coo[flag][l2];
						csr_e1[1-flag][i++] = csr_e1[flag][l2++];
					}
				}
			}				

			long weight=1;

			// long cq=0;
			long cr=0;

			// dense bit extract (and mcsr_e making)
			for(i=row_ptr[row_panel*BH]+1; i<row_ptr[(row_panel+1)*BH]; i++) {
				if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
				else {
					if(weight >= THRESHOLD) {
						cr++;
					} 				//if(cr == BW) { cq++; cr=0;}
					weight = 1;
				}
			}
			//long reminder = (csr_e1[flag][i-1]&31);
			if(weight >= THRESHOLD) {
				cr++;
			} 		//if(cr == BW) { cq++; cr=0; }
					// TODO = occ control
			mcsr_cnt[row_panel+1] = CEIL(cr,BW)+1;

		}

		// prefix-sum
		for(long i=1; i<=npanel;i++) 
			mcsr_cnt[i] += mcsr_cnt[i-1];
		//mcsr_e[0] = 0;
		mcsr_e[BH * mcsr_cnt[npanel]] = nnz;

		#pragma omp parallel for schedule(dynamic, 1)
		for(long row_panel=0; row_panel<m/BH; row_panel++) {
			long tid = omp_get_thread_num();
			if(mcsr_chk[row_panel] == 0) {
				long i, j;
				long flag = 0;
				long cq=0, cr=0;
				for(long stride = 1; stride <= BH/2; stride*=2, flag=1-flag);
				long base = (mcsr_cnt[row_panel]*BH);
				long mfactor = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
				long weight=1;

				// mcsr_e making
				for(i=row_ptr[row_panel*BH]+1; i<row_ptr[(row_panel+1)*BH]; i++) {
					if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
					else {
						long reminder = (csr_e1[flag][i-1]&31);
						if(weight >= THRESHOLD) {
							cr++;
							bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder); 
							for(j=i-weight; j<=i-1; j++) {
								mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
							}
						} else {
							//bv[tid][csr_e1[flag][i-1]>>5] &= (~0 - (1<<reminder)); 
							bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder)); 
						} 
						if(cr == BW) { cq++; cr=0;}
						weight = 1;
					}
				}

				//fprintf(stderr, "inter : %ld\n", i);

				long reminder = (csr_e1[flag][i-1]&31);
				if(weight >= THRESHOLD) {
					cr++;
					bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder); 
					for(j=i-weight; j<=i-1; j++) {
						mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
					}
				} else {
					bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder)); 
				} 
				// reordering
				long delta = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
				long base0 = mcsr_cnt[row_panel]*BH;
				for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
					long base = base0+(i-row_panel*BH)*delta;
					long dpnt = mcsr_e[base] = row_ptr[i];
					for(long j=1;j<delta;j++) {
						mcsr_e[base+j] += mcsr_e[base+j-1];
					}
					long spnt=mcsr_e[mcsr_cnt[row_panel]*BH + (mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel])*(i - row_panel*BH + 1) - 1];

					avg0[tid] += row_ptr[i+1] - spnt; 
					for(j=row_ptr[i]; j<row_ptr[i+1]; j++) {
						long k = col_ind[j];
						if((bv[tid][k>>5]&(1<<(k&31)))) {
							ja[dpnt] = col_ind[j];
							a[dpnt++] = values[j];
						} else {
							ja[spnt] = col_ind[j];
							a[spnt++] = values[j];
						}
					}
				}
			} else {
				long base0 = mcsr_cnt[row_panel]*BH;
				memcpy(&mcsr_e[base0], &row_ptr[row_panel*BH], sizeof(INT_T)*BH);
				avg0[tid] += row_ptr[(row_panel+1)*BH] - row_ptr[row_panel*BH];
				long bidx = row_ptr[row_panel*BH];
				long bseg = row_ptr[(row_panel+1)*BH] - bidx;
				memcpy(&ja[bidx], &col_ind[bidx], sizeof(INT_T)*bseg);
				memcpy(&a[bidx], &values[bidx], sizeof(ValueType)*bseg);

			}
		}



		double avg = 0;
		for(long i=0;i<num_threads;i++) 
			avg += avg0[i];
		avg /= (double)m;

		special_p = 0;
		for(long i=0;i<m;i++) {
			long idx = (mcsr_cnt[i>>LOG_BH])*BH + (mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH])*((i&(BH-1))+1);
			long diff = row_ptr[i+1] - mcsr_e[idx-1]; 
			double r = ((double)diff - avg);
			vari += r * r;

			if(diff >= STHRESHOLD) {
				long pp = (diff) / STHRESHOLD;
				for(long j=0; j<pp; j++) {
					special[special_p] = i;
					special2[special_p] = j * STHRESHOLD;
					special_p++;
				}
			}



		}
		vari /= (double)m;


		for(long i=0;i<num_threads;i++)
			free(bv[i]);
		for(long i=0;i<2;i++) {
			free(csr_e1[i]);
			free(coo[i]);
		}

		free(bv);
		free(csr_e1);
		free(coo);	

	}

	~ASpT()
	{
		free(a);
		free(row_ptr);
		free(ja);
	}

	void sddmm(long K, ValueType * A, ValueType * B, ValueType * C);
	void statistics_start();
	int statistics_print_data(char * buf, long buf_n);
};


void
ASpT::sddmm(long K, ValueType * A, ValueType * B, ValueType * C)
{
	row_ptr = (typeof(row_ptr)) __builtin_assume_aligned(row_ptr, 64);
	ja = (typeof(ja)) __builtin_assume_aligned(ja, 64);
	a = (typeof(a)) __builtin_assume_aligned(a, 64);
	C = (typeof(C)) __builtin_assume_aligned(C, 64);
	//__builtin_assume_aligned(mcsr_cnt, 64);
	//__builtin_assume_aligned(mcsr_e, 64);
	//__builtin_assume_aligned(mcsr_list, 64);
	A = (typeof(A)) __builtin_assume_aligned(A, 64);
	B = (typeof(B)) __builtin_assume_aligned(B, 64);

	#pragma omp parallel for
	for (long i=0;i<nnz;i++)
	{
		C[i] = 0;
	}


	if(vari < 5000*1/1*1)
	{
		// #pragma ivdep
		// #pragma vector aligned
		// #pragma temporal (A)
		#pragma omp parallel for schedule(dynamic, 1)
		for(long row_panel=0; row_panel<m/BH; row_panel ++) {
			//dense
			long stride;
			for(stride = 0; stride < mcsr_cnt[row_panel+1]-mcsr_cnt[row_panel]-1; stride++) {

				for(long i=row_panel*BH; i<(row_panel+1)*BH; i++) {
					long dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
					long loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

					long interm = loc1 + (((loc2 - loc1)>>3)<<3);
					long j;
					for(j=loc1; j<interm; j+=8) {
						//#pragma GCC ivdep
						//#pragma vector nontemporal (a)
						// #pragma prefetch A:_MM_HINT_T1
						// #pragma temporal (A)
						for(long k=0; k<K; k++) {
							// __builtin_prefetch(&A[ja[j+8]*K + k], 2);
							C[j]   += A[ja[j]*K + k] * B[i*K + k];
							C[j+1] += A[ja[j+1]*K + k] * B[i*K + k];
							C[j+2] += A[ja[j+2]*K + k] * B[i*K + k];
							C[j+3] += A[ja[j+3]*K + k] * B[i*K + k];
							C[j+4] += A[ja[j+4]*K + k] * B[i*K + k];
							C[j+5] += A[ja[j+5]*K + k] * B[i*K + k];
							C[j+6] += A[ja[j+6]*K + k] * B[i*K + k];
							C[j+7] += A[ja[j+7]*K + k] * B[i*K + k];
						}
						#pragma GCC ivdep
						for(long k=0; k<8; k++) {
							C[j+k] *= a[j+k];
						}

					}
					for(; j<loc2; j++) {
						//#pragma GCC ivdep
						//#pragma vector nontemporal (a)
						// #pragma prefetch A:_MM_HINT_T1
						// #pragma temporal (A)
						for(long k=0; k<K; k++) {
							C[j] += A[ja[j]*K + k] * B[i*K + k];
						}
						C[j] *= a[j];
					}
				}

			}
			//sparse
			for(long i=row_panel*BH; i<(row_panel+1)*BH; i++) {

				long dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
				long loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

				//printf("(%ld %ld %ld %ld %ld)\n", i, row_ptr[i], loc1, row_ptr[i+1], loc2);
				//printf("%ld %ld %ld %ld %ld %ld %ld\n", i, dummy, stride, row_ptr[i], loc1, row_ptr[i+1], loc2);


				long interm = loc1 + (((loc2 - loc1)>>3)<<3);
				long j;
				for(j=loc1; j<interm; j+=8) {
					//#pragma GCC ivdep
					//#pragma vector nontemporal (a)
					// #pragma prefetch A:_MM_HINT_T1
					// #pragma temporal (A)
					for(long k=0; k<K; k++) {
						C[j]   += A[ja[j]*K + k] * B[i*K + k];
						C[j+1] += A[ja[j+1]*K + k] * B[i*K + k];
						C[j+2] += A[ja[j+2]*K + k] * B[i*K + k];
						C[j+3] += A[ja[j+3]*K + k] * B[i*K + k];
						C[j+4] += A[ja[j+4]*K + k] * B[i*K + k];
						C[j+5] += A[ja[j+5]*K + k] * B[i*K + k];
						C[j+6] += A[ja[j+6]*K + k] * B[i*K + k];
						C[j+7] += A[ja[j+7]*K + k] * B[i*K + k];
					} 
					#pragma GCC ivdep
					for(long k=0; k<8; k++) {
						C[j+k] *= a[j+k];
					}
				}
				for(; j<loc2; j++) {
					//#pragma GCC ivdep
					//#pragma vector nontemporal (a)
					// #pragma prefetch A:_MM_HINT_T1
					// #pragma temporal (A)
					for(long k=0; k<K; k++) {
						C[j] = C[j] + A[ja[j]*K + k] * B[i*K + k];
					} 
					C[j] *= a[j];
				}
			}
		}
	}
	else // big var
	{
		// #pragma ivdep
		// #pragma vector aligned
		// #pragma temporal (A)
		#pragma omp parallel for schedule(dynamic, 1)
		for(long row_panel=0; row_panel<m/BH; row_panel ++)
		{
			//dense
			long stride;
			for(stride = 0; stride < mcsr_cnt[row_panel+1]-mcsr_cnt[row_panel]-1; stride++) {

				for(long i=row_panel*BH; i<(row_panel+1)*BH; i++) {
					long dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
					long loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

					long interm = loc1 + (((loc2 - loc1)>>3)<<3);
					long j;
					for(j=loc1; j<interm; j+=8) {
						//#pragma ivdep
						//#pragma vector nontemporal (a)
						// #pragma prefetch A:_MM_HINT_T1
						// #pragma temporal (A)
						for(long k=0; k<K; k++) {
							C[j] = C[j] + A[ja[j]*K + k] * B[i*K + k];
							C[j+1] = C[j+1] + A[ja[j+1]*K + k] * B[i*K + k];
							C[j+2] = C[j+2] + A[ja[j+2]*K + k] * B[i*K + k];
							C[j+3] = C[j+3] + A[ja[j+3]*K + k] * B[i*K + k];
							C[j+4] = C[j+4] + A[ja[j+4]*K + k] * B[i*K + k];
							C[j+5] = C[j+5] + A[ja[j+5]*K + k] * B[i*K + k];
							C[j+6] = C[j+6] + A[ja[j+6]*K + k] * B[i*K + k];
							C[j+7] = C[j+7] + A[ja[j+7]*K + k] * B[i*K + k];
						} 
						#pragma GCC ivdep
						for(long k=0; k<8; k++) {
							C[j+k] *= a[j+k];
						}
					}
					for(; j<loc2; j++) {
						//#pragma GCC ivdep
						//#pragma vector nontemporal (a)
						// #pragma prefetch A:_MM_HINT_T1
						// #pragma temporal (A)
						for(long k=0; k<K; k++) {
							C[j] = C[j] + A[ja[j]*K + k] * B[i*K + k];
						}	 
						C[j] *= a[j];
					}
				}

			}
			//sparse
			for(long i=row_panel*BH; i<(row_panel+1)*BH; i++) {

				long dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
				long loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

				loc1 += ((loc2 - loc1)/STHRESHOLD)*STHRESHOLD;

				long interm = loc1 + (((loc2 - loc1)>>3)<<3);
				long j;
				for(j=loc1; j<interm; j+=8) {
					//#pragma GCC ivdep
					//#pragma vector nontemporal (a)
					// #pragma prefetch A:_MM_HINT_T1
					// #pragma temporal (A)
					for(long k=0; k<K; k++) {
						C[j] = C[j] + A[ja[j]*K + k] * B[i*K + k];
						C[j+1] = C[j+1] + A[ja[j+1]*K + k] * B[i*K + k];
						C[j+2] = C[j+2] + A[ja[j+2]*K + k] * B[i*K + k];
						C[j+3] = C[j+3] + A[ja[j+3]*K + k] * B[i*K + k];
						C[j+4] = C[j+4] + A[ja[j+4]*K + k] * B[i*K + k];
						C[j+5] = C[j+5] + A[ja[j+5]*K + k] * B[i*K + k];
						C[j+6] = C[j+6] + A[ja[j+6]*K + k] * B[i*K + k];
						C[j+7] = C[j+7] + A[ja[j+7]*K + k] * B[i*K + k];
					} 
					#pragma GCC ivdep
					for(long k=0; k<8; k++) {
						C[j+k] *= a[j+k];
					}
				}
				for(; j<loc2; j++) {
					//#pragma GCC ivdep
					//#pragma vector nontemporal (a)
					// #pragma prefetch A:_MM_HINT_T1
					// #pragma temporal (A)
					for(long k=0; k<K; k++) {
						C[j] = C[j] + A[ja[j]*K + k] * B[i*K + k];
					} 
					C[j] *= a[j];
				}
			}
		}
		// #pragma ivdep
		// #pragma vector aligned
		// #pragma temporal (A)
		#pragma omp parallel for schedule(dynamic, 1)
		for(long row_panel=0; row_panel<special_p;row_panel ++)
		{
			long i=special[row_panel];

			long dummy = mcsr_cnt[i>>LOG_BH]*BH + ((i&(BH-1))+1)*(mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH]);

			long loc1 = mcsr_e[dummy-1] + special2[row_panel];
			long loc2 = loc1 + STHRESHOLD;

			//long interm = loc1 + (((loc2 - loc1)>>3)<<3);
			long j;
			//assume to 128
			//ValueType temp_r[128]={0,};
			//for(long e=0;e<128;e++) {
			//	temp_r[e] = 0.0f;
			//}

			for(j=loc1; j<loc2; j+=8) {
				//#pragma GCC ivdep
				//#pragma vector nontemporal (a)
				// #pragma prefetch A:_MM_HINT_T1
				// #pragma temporal (A)
				for(long k=0; k<K; k++) {
					C[j] = C[j] + A[ja[j]*K + k] * B[i*K + k];
					C[j+1] = C[j+1] + A[ja[j+1]*K + k] * B[i*K + k];
					C[j+2] = C[j+2] + A[ja[j+2]*K + k] * B[i*K + k];
					C[j+3] = C[j+3] + A[ja[j+3]*K + k] * B[i*K + k];
					C[j+4] = C[j+4] + A[ja[j+4]*K + k] * B[i*K + k];
					C[j+5] = C[j+5] + A[ja[j+5]*K + k] * B[i*K + k];
					C[j+6] = C[j+6] + A[ja[j+6]*K + k] * B[i*K + k];
					C[j+7] = C[j+7] + A[ja[j+7]*K + k] * B[i*K + k];
				} 
				#pragma GCC ivdep
				for(long k=0; k<8; k++) {
					C[j+k] *= a[j+k];
				}
			}
		}
	} 

	// #define VALIDATE
	#if defined VALIDATE
		ValueType *vout_gold;
		vout_gold = (ValueType *)malloc(sizeof(ValueType)*nnz+256);
		memset(vout_gold, 0, sizeof(ValueType)*nnz+256);
		//validate
		for(long i=0;i<nnz;i++) {
			vout_gold[i] = 0.0f;
		}
		for(long i=0;i<m;i++) {
			for(long j=row_ptr[i]; j<row_ptr[i+1]; j++) {
				for(long k=0; k<K; k++) {
					vout_gold[j] += A[K*ja[j] + k] * B[K*i + k];
				}
				vout_gold[j] *= a[j];
			}
		}
		long num_diff=0;
		for(long i=0;i<nnz;i++) {
			ValueType p1 = vout_gold[i];
			ValueType p2 = C[i];

			if(p1 < 0)
				p1 *= -1;
			if(p2 < 0)
				p2 *= -1;
			ValueType diff;
			diff = p1 - p2;
			if(diff < 0)
				diff *= -1;
			if(diff / MAX(p1,p2) > 0.01) {
				//if(num_diff < 20*1*1) fprintf(stdout, "%d %f %f\n", i, B[i], vout_gold[i]);
				//if(B[i] < vout_gold[i]) fprintf(stdout, "%d %f %f\n", i, B[i], vout_gold[i]);

				num_diff++;
			}
		}
		//      fprintf(stdout, "num_diff : %d\n", num_diff);
		// fprintf(stdout, "%f\n", (double)num_diff/nnz*100);
		fprintf(stdout, "num errors: %ld\n", num_diff);
	#endif

}


struct Matrix_Format *
csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueTypeReference * values, long m, long n, long nnz, long symmetric, long symmetry_expanded)
{
	if (symmetric && !symmetry_expanded)
		error("symmetric matrices have to be expanded to be supported by this format");
	struct ASpT * csr = new ASpT(row_ptr, col_ind, values, m, n, nnz);
	csr->mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	csr->format_name = (char *) "Custom_CSR";
	return csr;
}


//==========================================================================================================================================
//= Print Statistics
//==========================================================================================================================================


void
ASpT::statistics_start()
{
}


int
statistics_print_labels(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}


int
ASpT::statistics_print_data(__attribute__((unused)) char * buf, __attribute__((unused)) long buf_n)
{
	return 0;
}

