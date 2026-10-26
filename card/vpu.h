/*
 * vpu.h: KNC F32 vector helpers on the S3 clang builtins (BuiltinsX86KNC.td). Every function is vector instructions
 * only. Precision per Intel 327364-001: vexp223ps 0.99 ULP on its Q8.24 input (p.190), vrcp23ps 0.912 ULP (p.577),
 * vrsqrt23ps 0.775 ULP (p.588), vlog2ps 4 ULP outside [0.5, 2] and absolute < 2^-21 inside (p.373).
 */
#ifndef CX_VPU_H
#define CX_VPU_H

#include <immintrin.h>

typedef float v16f __attribute__((vector_size(64)));
typedef int v16i __attribute__((vector_size(64)));

#define CX_ALL ((unsigned short)0xffff)

/* vcvtfxpntps2dq imm8 (327364 p.169): I1:I0 rounding (00 nearest even), I7:I4 exponent adjustment (0101 = 2^24, so
 * the result is fixed point 8.24, the input format of vexp223ps) */
#define CX_FXP_8_24_RN 0x50

/* 2^x: x to Q8.24 (saturating: below -128 gives INT_MIN -> +0, at or above 128 INT_MAX -> +inf), then vexp223ps */
static inline __m512 cx_exp2(__m512 x)
{
	const v16i q = __builtin_ia32_knc_vcvtfxpntps2dq_zzi((v16i)_mm512_undefined_epi32(), CX_ALL, (v16f)x,
	                                                     CX_FXP_8_24_RN, 0);
	return (__m512)__builtin_ia32_knc_vexp223ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)q, 0);
}

static inline __m512 cx_rcp(__m512 x)
{
	return (__m512)__builtin_ia32_knc_vrcp23ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)x, 0);
}

static inline __m512 cx_rsqrt(__m512 x)
{
	return (__m512)__builtin_ia32_knc_vrsqrt23ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)x, 0);
}

static inline __m512 cx_log2(__m512 x)
{
	return (__m512)__builtin_ia32_knc_vlog2ps_zz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)x, 0);
}

#endif
