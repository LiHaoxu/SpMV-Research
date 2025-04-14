#ifndef VECTORIZATION_X86_AVX512_M32_H
#define VECTORIZATION_X86_AVX512_M32_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>

#include "macros/cpp_defines.h"
#include "macros/macrolib.h"

#include "vectorization/vectorization_util.h"


typedef union __attribute__((packed)) { __mmask8  v;    }  vec_mask_m32_1_t;
typedef union __attribute__((packed)) { __mmask8  v;    }  vec_mask_m32_4_t;
typedef union __attribute__((packed)) { __mmask8  v;    }  vec_mask_m32_8_t;
typedef union __attribute__((packed)) { __mmask16 v;    }  vec_mask_m32_16_t;
typedef union __attribute__((packed)) { __mmask16 v[2]; }  vec_mask_m32_32_t;

typedef uint8_t  vec_mask_packed_m32_1_t;
typedef uint8_t  vec_mask_packed_m32_4_t;
typedef uint8_t  vec_mask_packed_m32_8_t;
typedef uint16_t vec_mask_packed_m32_16_t;
typedef uint32_t vec_mask_packed_m32_32_t;


#define vec_mask_m32_16(val)                               ( (vec_mask_m32_16_t) { .v = (val) } )
#define vec_mask_m32_8(val)                                ( (vec_mask_m32_8_t)  { .v = (val) } )
#define vec_mask_m32_4(val)                                ( (vec_mask_m32_4_t)  { .v = (val) } )
#define vec_mask_m32_1(val)                                ( (vec_mask_m32_1_t)  { .v = (val) } )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Set - Load - Store
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_mask_pack_m32_32(a)                            vec_loop_expr_init(vec_mask_packed_m32_32_t, 2, _tmp, 0, _i, _tmp |= ((vec_mask_packed_m32_32_t) a.v[_i]) << (_i * 16);)
#define vec_mask_pack_m32_16(a)                            (a.v)
#define vec_mask_pack_m32_8(a)                             (a.v)
#define vec_mask_pack_m32_4(a)                             (a.v)
#define vec_mask_pack_m32_1(a)                             (a.v)

#define vec_mask_packed_get_bit_m32_32(a, pos)             bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_16(a, pos)             bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_8(a, pos)              bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_4(a, pos)              bits_u32_extract(a, pos, 1)
#define vec_mask_packed_get_bit_m32_1(a, pos)              bits_u32_extract(a, pos, 1)

#define vec_mask_set_m32_32(expr)                          vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, _tmp.v[_i] = _cvtu32_mask16(expr);)
#define vec_mask_set_m32_16(expr)                          vec_mask_m32_16( _cvtu32_mask16(expr) )
#define vec_mask_set_m32_8(expr)                           vec_mask_m32_8( _cvtu32_mask8(expr) )
#define vec_mask_set_m32_4(expr)                           vec_mask_m32_4( _cvtu32_mask8(expr) )
#define vec_mask_set_m32_1(expr)                           vec_mask_m32_1( (uint8_t) vec_iter_expr_1(_j, 0, ((expr) & (1 << _j)) ? 1 : 0) )


//------------------------------------------------------------------------------------------------------------------------------------------
//- Operations
//------------------------------------------------------------------------------------------------------------------------------------------

#define vec_and_m32_32(a, b)                               vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, _tmp.v[_i] = _mm512_kand(a.v[_i], b.v[_i]);)
#define vec_and_m32_16(a, b)                               vec_mask_m32_16( _kand_mask16(a.v, b.v) )
#define vec_and_m32_8(a, b)                                vec_mask_m32_8( _kand_mask8(a.v, b.v) )
#define vec_and_m32_4(a, b)                                vec_mask_m32_4( _kand_mask8(a.v, b.v) )
#define vec_and_m32_1(a, b)                                vec_mask_m32_1( (uint8_t) (a.v & b.v) )

#define vec_or_m32_32(a, b)                                vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, _tmp.v[_i] = _mm512_kor(a.v[_i], b.v[_i]);)
#define vec_or_m32_16(a, b)                                vec_mask_m32_16( _kor_mask16(a.v, b.v) )
#define vec_or_m32_8(a, b)                                 vec_mask_m32_8( _kor_mask8(a.v, b.v) )
#define vec_or_m32_4(a, b)                                 vec_mask_m32_4( _kor_mask8(a.v, b.v) )
#define vec_or_m32_1(a, b)                                 vec_mask_m32_1( (uint8_t) (a.v | b.v) )

#define vec_not_m32_32(a)                                  vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, _tmp.v[_i] = _mm512_knot(a.v[_i]);)
#define vec_not_m32_16(a)                                  vec_mask_m32_16( _knot_mask16(a.v) )
#define vec_not_m32_8(a)                                   vec_mask_m32_8( _knot_mask8(a.v) )
#define vec_not_m32_4(a)                                   vec_mask_m32_4( _knot_mask8(a.v) )
#define vec_not_m32_1(a)                                   vec_mask_m32_1( (uint8_t) (a.v ^ 1) )

#define vec_xor_m32_32(a, b)                               vec_loop_expr(vec_mask_m32_32_t, 4, _tmp, _i, _tmp.v[_i] = _mm512_kxor(a.v[_i], b.v[_i]);)
#define vec_xor_m32_16(a, b)                               vec_mask_m32_16( _kxor_mask16(a.v, b.v) )
#define vec_xor_m32_8(a, b)                                vec_mask_m32_8( _kxor_mask8(a.v, b.v) )
#define vec_xor_m32_4(a, b)                                vec_mask_m32_4( _kxor_mask8(a.v, b.v) )
#define vec_xor_m32_1(a, b)                                vec_mask_m32_1( (uint8_t) (a.v ^ b.v) )


#endif /* VECTORIZATION_X86_AVX512_M32_H */

