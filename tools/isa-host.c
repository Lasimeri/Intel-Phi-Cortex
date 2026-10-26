/*
 * isa-host: Stage 0c, host side. The host makes the inputs and judges the card's outputs against libm in double.
 *
 *   isa-host gen DIR     writes DIR/{exp2,rcp,rsqrt,log2,log2d}.in, 2^20 float32s each
 *   isa-host check DIR   reads each .in with its .out (from card/isa.c) and prints error statistics
 *
 * ULP of a reference r: 2^(e - 23) with e = floor(log2 |r|), at least 2^-149 (subnormal spacing).
 * Sets:
 *   exp2  - first 16 lanes are special values; then half uniform in [-125, 127], half uniform in [-1, 1]. Also judged
 *           against 2^(x rounded to Q8.24), which isolates vexp223ps from the input conversion.
 *   rcp   - |x| log-uniform in [2^-20, 2^20], random sign
 *   rsqrt - x log-uniform in [2^-20, 2^20]
 *   log2  - x log-uniform in [2^-20, 2^20]; ULP judged only outside [0.5, 2] (the manual's 4 ULP range)
 *   log2d - x uniform in [0.5, 2]; judged by absolute error (manual: below 2^-21)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (1 << 20)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double uni(void)   /* xorshift64*, [0, 1) */
{
	rng ^= rng >> 12;
	rng ^= rng << 25;
	rng ^= rng >> 27;
	return (double)((rng * 0x2545F4914F6CDD1Dull) >> 11) / 9007199254740992.0;
}

static const float special[16] = { -INFINITY, -0.0f, 0.0f, NAN, 1, 2, -1, -2, 10, -10, -126, 127, -200, 200, 0.5f, -0.5f };

static int put(const char *dir, const char *name, const float *v)
{
	char path[4096];
	snprintf(path, sizeof path, "%s/%s.in", dir, name);
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(v, sizeof *v, N, f) != N || fclose(f)) { perror(path); return -1; }
	return 0;
}

static float *get(const char *dir, const char *name, const char *ext)
{
	char path[4096];
	snprintf(path, sizeof path, "%s/%s.%s", dir, name, ext);
	FILE *f = fopen(path, "rb");
	float *v = malloc(N * sizeof *v);
	if (!f || fread(v, sizeof *v, N, f) != N) { perror(path); exit(1); }
	fclose(f);
	return v;
}

static double ulp_of(double r)
{
	if (r == 0 || !isfinite(r)) return ldexp(1, -149);
	int k;
	frexp(r, &k);   /* |r| in [2^(k-1), 2^k) */
	const int e = k - 1 < -126 ? -126 : k - 1;
	return ldexp(1, e - 23);
}

static int cmpd(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return (x > y) - (x < y);
}

/* error statistics over err[0..n): max, median, p99.9, count above lim */
static void report(const char *what, double *err, long n, double lim, const char *unit)
{
	qsort(err, (size_t)n, sizeof *err, cmpd);
	long over = 0;
	for (long i = 0; i < n; i++) over += err[i] > lim;
	printf("%-28s n %7ld  max %.4g  median %.4g  p99.9 %.4g  > %g %s: %ld\n", what, n, err[n - 1], err[n / 2],
	       err[(long)(n * 0.999)], lim, unit, over);
}

static int gen(const char *dir)
{
	float *v = malloc(N * sizeof *v);
	memcpy(v, special, sizeof special);
	for (long i = 16; i < N; i++) v[i] = (float)(i < N / 2 ? -125 + 252 * uni() : -1 + 2 * uni());
	if (put(dir, "exp2", v)) return 1;
	for (long i = 0; i < N; i++) v[i] = (float)(exp2(-20 + 40 * uni()) * (uni() < 0.5 ? -1 : 1));
	if (put(dir, "rcp", v)) return 1;
	for (long i = 0; i < N; i++) v[i] = (float)exp2(-20 + 40 * uni());
	if (put(dir, "rsqrt", v) || put(dir, "log2", v)) return 1;
	for (long i = 0; i < N; i++) v[i] = (float)(0.5 + 1.5 * uni());
	if (put(dir, "log2d", v)) return 1;
	printf("gen: 5 sets of %d float32s in %s\n", N, dir);
	return 0;
}

static int check(const char *dir)
{
	double *e = malloc(N * sizeof *e), *e2 = malloc(N * sizeof *e2);
	long n, n2;

	float *x = get(dir, "exp2", "in"), *y = get(dir, "exp2", "out");
	printf("exp2 specials:");
	for (int i = 0; i < 16; i++) printf(" 2^%g=%g", special[i], y[i]);
	printf("\n");
	n = 0;
	for (long i = 16; i < N; i++) {
		const double r = exp2((double)x[i]), rq = exp2(nearbyint((double)x[i] * 16777216.0) / 16777216.0);
		e[n] = fabs(y[i] - r) / ulp_of(r);
		e2[n++] = fabs(y[i] - rq) / ulp_of(rq);
	}
	report("exp2 vs 2^x", e, n, 1, "ulp");
	report("exp2 vs 2^(Q8.24 x)", e2, n, 0.99, "ulp");
	free(x); free(y);

	x = get(dir, "rcp", "in"); y = get(dir, "rcp", "out");
	for (long i = 0; i < N; i++) { const double r = 1.0 / x[i]; e[i] = fabs(y[i] - r) / ulp_of(r); }
	report("rcp vs 1/x", e, N, 0.912, "ulp");
	free(x); free(y);

	x = get(dir, "rsqrt", "in"); y = get(dir, "rsqrt", "out");
	for (long i = 0; i < N; i++) { const double r = 1.0 / sqrt((double)x[i]); e[i] = fabs(y[i] - r) / ulp_of(r); }
	report("rsqrt vs 1/sqrt(x)", e, N, 0.775, "ulp");
	free(x); free(y);

	x = get(dir, "log2", "in"); y = get(dir, "log2", "out");
	n = n2 = 0;
	for (long i = 0; i < N; i++) {
		const double r = log2((double)x[i]);
		if (x[i] < 0.5f || x[i] > 2.0f) e[n++] = fabs(y[i] - r) / ulp_of(r);
		else e2[n2++] = fabs(y[i] - r);
	}
	report("log2 outside [0.5,2]", e, n, 4, "ulp");
	report("log2 inside [0.5,2] abs", e2, n2, ldexp(1, -21), "abs");
	free(x); free(y);

	x = get(dir, "log2d", "in"); y = get(dir, "log2d", "out");
	for (long i = 0; i < N; i++) e[i] = fabs(y[i] - log2((double)x[i]));
	report("log2d [0.5,2] abs", e, N, ldexp(1, -21), "abs");
	free(x); free(y);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 3 && !strcmp(argv[1], "gen")) return gen(argv[2]);
	if (argc == 3 && !strcmp(argv[1], "check")) return check(argv[2]);
	fprintf(stderr, "usage: isa-host gen|check DIR\n");
	return 2;
}
