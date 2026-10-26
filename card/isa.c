/*
 * isa: Stage 0c, card side. usage: isa OP IN OUT, OP one of exp2 rcp rsqrt log2. Reads float32s from IN (a multiple
 * of 16), applies the op 16 lanes at a time on the vector unit (vpu.h), writes the results to OUT. The host makes the
 * inputs and judges the outputs (tools/isa-host.c); the card does no scalar math.
 */
#include "vpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
	if (argc != 4) { fprintf(stderr, "usage: isa exp2|rcp|rsqrt|log2 IN OUT\n"); return 2; }
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	fseek(f, 0, SEEK_END);
	const long bytes = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (bytes <= 0 || bytes % 64) { fprintf(stderr, "isa: %s is %ld bytes, not a multiple of 64\n", argv[2], bytes); return 1; }
	float *buf = aligned_alloc(64, (size_t)bytes);
	if (!buf || fread(buf, 1, (size_t)bytes, f) != (size_t)bytes) { fprintf(stderr, "isa: read %s\n", argv[2]); return 1; }
	fclose(f);

	const char *op = argv[1];
	const long n = bytes / 4;
	for (long i = 0; i < n; i += 16) {
		const __m512 x = _mm512_load_ps(buf + i);
		__m512 y;
		if (!strcmp(op, "exp2")) y = cx_exp2(x);
		else if (!strcmp(op, "rcp")) y = cx_rcp(x);
		else if (!strcmp(op, "rsqrt")) y = cx_rsqrt(x);
		else if (!strcmp(op, "log2")) y = cx_log2(x);
		else { fprintf(stderr, "isa: unknown op %s\n", op); return 2; }
		_mm512_store_ps(buf + i, y);
	}

	f = fopen(argv[3], "wb");
	if (!f || fwrite(buf, 1, (size_t)bytes, f) != (size_t)bytes || fclose(f)) { perror(argv[3]); return 1; }
	printf("isa %s: %ld values\n", op, n);
	return 0;
}
