#ifndef VECTORIZATION_RISCV_SVV_M32_H
#define VECTORIZATION_RISCV_SVV_M32_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <riscv_vector.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"


typedef int32_t   __attribute__((may_alias))  vec_mask_m32_1_t;
typedef vbool32_t __attribute__((may_alias))  vec_mask_m32_4_t;
typedef vbool32_t __attribute__((may_alias))  vec_mask_m32_8_t;
typedef vbool32_t __attribute__((may_alias))  vec_mask_m32_16_t;
typedef vbool32_t __attribute__((may_alias))  vec_mask_m32_32_t;

typedef uint32_t vec_mask_packed_m32_1_t;
typedef uint32_t vec_mask_packed_m32_4_t;
typedef uint32_t vec_mask_packed_m32_8_t;
typedef uint32_t vec_mask_packed_m32_16_t;
typedef uint32_t vec_mask_packed_m32_32_t;


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_elem_get_m32_32(vec, index)                    ({int32_t _buf[32] = {0}; __riscv_vsm_v_b32((uint8_t *) _buf, vec, 32); _buf[index];})
#define vec_elem_get_m32_16(vec, index)                    ({int32_t _buf[16] = {0}; __riscv_vsm_v_b32((uint8_t *) _buf, vec, 16); _buf[index];})
#define vec_elem_get_m32_8(vec, index)                     ({int32_t _buf[ 8] = {0}; __riscv_vsm_v_b32((uint8_t *) _buf, vec,  8); _buf[index];})
#define vec_elem_get_m32_4(vec, index)                     ({int32_t _buf[ 4] = {0}; __riscv_vsm_v_b32((uint8_t *) _buf, vec,  4); _buf[index];})
#define vec_elem_get_m32_1(vec, index)                     vec

#define vec_elem_set_m32_32(vec, index, expr)              do { int32_t _buf[32]; __riscv_vsm_v_b32((uint8_t *) _buf, vec, 32); _buf[index] = (int32_t) (expr); vec = __riscv_vlm_v_b32((const uint8_t *) _buf, 32); } while (0)
#define vec_elem_set_m32_16(vec, index, expr)              do { int32_t _buf[16]; __riscv_vsm_v_b32((uint8_t *) _buf, vec, 16); _buf[index] = (int32_t) (expr); vec = __riscv_vlm_v_b32((const uint8_t *) _buf, 16); } while (0)
#define vec_elem_set_m32_8(vec, index, expr)               do { int32_t _buf[ 8]; __riscv_vsm_v_b32((uint8_t *) _buf, vec,  8); _buf[index] = (int32_t) (expr); vec = __riscv_vlm_v_b32((const uint8_t *) _buf,  8); } while (0)
#define vec_elem_set_m32_4(vec, index, expr)               do { int32_t _buf[ 4]; __riscv_vsm_v_b32((uint8_t *) _buf, vec,  4); _buf[index] = (int32_t) (expr); vec = __riscv_vlm_v_b32((const uint8_t *) _buf,  4); } while (0)
#define vec_elem_set_m32_1(vec, index, expr)               do { vec = (expr); } while (0)

#define vec_mask_pack_m32_32(a)                            ({int32_t _buf[32]; __riscv_vsm_v_b32((uint8_t *) &_buf, a, 32); _buf[0];})
#define vec_mask_pack_m32_16(a)                            ({int32_t _buf[16]; __riscv_vsm_v_b32((uint8_t *) &_buf, a, 16); _buf[0];})
#define vec_mask_pack_m32_8(a)                             ({int32_t _buf[ 8]; __riscv_vsm_v_b32((uint8_t *) &_buf, a,  8); _buf[0];})
#define vec_mask_pack_m32_4(a)                             ({int32_t _buf[ 4]; __riscv_vsm_v_b32((uint8_t *) &_buf, a,  4); _buf[0];})
#define vec_mask_pack_m32_1(a)                             a

#define vec_mask_packed_get_bit_m32_32(a, pos)             bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_16(a, pos)             bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_8(a, pos)              bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_4(a, pos)              bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_1(a, pos)              bits_u32_extract(a, pos, 1)

#define vec_mask_set_m32_32(expr)                          ({int32_t _buf[32] = {0}; _buf[0] = (expr) & (          - 1); __riscv_vlm_v_b32((const uint8_t *) _buf, 32);})
#define vec_mask_set_m32_16(expr)                          ({int32_t _buf[16] = {0}; _buf[0] = (expr) & ((1 << 16) - 1); __riscv_vlm_v_b32((const uint8_t *) _buf, 16);})
#define vec_mask_set_m32_8(expr)                           ({int32_t _buf[ 8] = {0}; _buf[0] = (expr) & ((1 <<  8) - 1); __riscv_vlm_v_b32((const uint8_t *) _buf,  8);})
#define vec_mask_set_m32_4(expr)                           ({int32_t _buf[ 4] = {0}; _buf[0] = (expr) & ((1 <<  4) - 1); __riscv_vlm_v_b32((const uint8_t *) _buf,  4);})
#define vec_mask_set_m32_1(expr)                           ( (int64_t) (expr) )

#define vec_mask_whilelt_m32_32(i, N)                      __riscv_whilelt(i, N, 32)
#define vec_mask_whilelt_m32_16(i, N)                      __riscv_whilelt(i, N, 16)
#define vec_mask_whilelt_m32_8(i, N)                       __riscv_whilelt(i, N,  8)
#define vec_mask_whilelt_m32_4(i, N)                       __riscv_whilelt(i, N,  4)
#define vec_mask_whilelt_m32_1(i, N)                       ( (uint8_t) ((i < N) ? -1 : 0) )

#define vec_mask_firstN_m32_32(N)                          vec_loop_expr(vec_mask_m32_32_t, 2, _tmp, _i, _tmp[_i] = (uint16_t) (((1ULL << (N)) - 1) >> 16*_i);)
#define vec_mask_firstN_m32_16(N)                          vec_mask_m32_16( (uint16_t) ((1ULL << (N)) - 1) )
#define vec_mask_firstN_m32_8(N)                           vec_mask_m32_8( (uint8_t) ((1ULL << (N)) - 1) )
#define vec_mask_firstN_m32_4(N)                           vec_mask_m32_4( (uint8_t) ((1ULL << (N)) - 1) )
#define vec_mask_firstN_m32_1(N)                           ( (uint8_t) ((1ULL << (N)) - 1) )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_and_m32_32(a, b)                               __riscv_vmand_mm_b32(a, b, 32)
#define vec_and_m32_16(a, b)                               __riscv_vmand_mm_b32(a, b, 16)
#define vec_and_m32_8(a, b)                                __riscv_vmand_mm_b32(a, b,  8)
#define vec_and_m32_4(a, b)                                __riscv_vmand_mm_b32(a, b,  4)
#define vec_and_m32_1(a, b)                                ( (uint8_t) (a & b) )

#define vec_or_m32_32(a, b)                                __riscv_vmor_mm_b32(a, b, 32)
#define vec_or_m32_16(a, b)                                __riscv_vmor_mm_b32(a, b, 16)
#define vec_or_m32_8(a, b)                                 __riscv_vmor_mm_b32(a, b,  8)
#define vec_or_m32_4(a, b)                                 __riscv_vmor_mm_b32(a, b,  4)
#define vec_or_m32_1(a, b)                                 ( (uint8_t) (a | b) )

#define vec_not_m32_32(a)                                  __riscv_vmnot_m_b32(a, 32)
#define vec_not_m32_16(a)                                  __riscv_vmnot_m_b32(a, 16)
#define vec_not_m32_8(a)                                   __riscv_vmnot_m_b32(a,  8)
#define vec_not_m32_4(a)                                   __riscv_vmnot_m_b32(a,  4)
#define vec_not_m32_1(a)                                   ( (uint8_t) (a ^ 1) )

#define vec_xor_m32_32(a, b)                               __riscv_vmxor_mm_b32(a, b, 32)
#define vec_xor_m32_16(a, b)                               __riscv_vmxor_mm_b32(a, b, 16)
#define vec_xor_m32_8(a, b)                                __riscv_vmxor_mm_b32(a, b,  8)
#define vec_xor_m32_4(a, b)                                __riscv_vmxor_mm_b32(a, b,  4)
#define vec_xor_m32_1(a, b)                                ( (uint8_t) (a ^ b) )


#endif /* VECTORIZATION_RISCV_SVV_M32_H */

