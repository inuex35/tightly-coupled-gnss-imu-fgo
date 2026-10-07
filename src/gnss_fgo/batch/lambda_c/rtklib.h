/* Minimal stand-in for RTKLIB's rtklib.h: only what lambda.c needs.
 * Matrices are column-major (RTKLIB convention). */
#ifndef CLAMBDA_RTKLIB_H
#define CLAMBDA_RTKLIB_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double *mat(int n, int m)
{
    if (n <= 0 || m <= 0) return NULL;
    return (double *)malloc(sizeof(double) * n * m);
}

static double *zeros(int n, int m)
{
    if (n <= 0 || m <= 0) return NULL;
    return (double *)calloc((size_t)n * m, sizeof(double));
}

static double *eye(int n)
{
    double *p = zeros(n, n);
    int i;
    if (p) for (i = 0; i < n; i++) p[i + i * n] = 1.0;
    return p;
}

/* C = alpha*op(A)*op(B) + beta*C, op = "NN","NT","TN","TT";
 * op(A) is n x m, op(B) is m x k, C is n x k. */
static void matmul(const char *tr, int n, int k, int m, double alpha,
                   const double *A, const double *B, double beta, double *C)
{
    int i, j, x;
    for (i = 0; i < n; i++) for (j = 0; j < k; j++) {
        double d = 0.0;
        for (x = 0; x < m; x++) {
            double a = tr[0] == 'N' ? A[i + x * n] : A[x + i * m];
            double b = tr[1] == 'N' ? B[x + j * m] : B[j + x * k];
            d += a * b;
        }
        C[i + j * n] = beta == 0.0 ? alpha * d : alpha * d + beta * C[i + j * n];
    }
}

/* X = op(A)\Y with A n x n, Y/X n x m; LU with partial pivoting. */
static int solve(const char *tr, const double *A, const double *Y, int n,
                 int m, double *X)
{
    double *B = mat(n, n);
    int *piv = (int *)malloc(sizeof(int) * n), i, j, k, info = 0;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++)
        B[i + j * n] = tr[0] == 'N' ? A[i + j * n] : A[j + i * n];
    for (k = 0; k < n; k++) {
        int p = k;
        for (i = k + 1; i < n; i++)
            if (fabs(B[i + k * n]) > fabs(B[p + k * n])) p = i;
        piv[k] = p;
        if (B[p + k * n] == 0.0) { info = -1; break; }
        if (p != k) for (j = 0; j < n; j++) {
            double t = B[k + j * n]; B[k + j * n] = B[p + j * n]; B[p + j * n] = t;
        }
        for (i = k + 1; i < n; i++) {
            B[i + k * n] /= B[k + k * n];
            for (j = k + 1; j < n; j++) B[i + j * n] -= B[i + k * n] * B[k + j * n];
        }
    }
    if (!info) {
        memcpy(X, Y, sizeof(double) * n * m);
        for (j = 0; j < m; j++) {
            double *x = X + j * n;
            for (k = 0; k < n; k++) if (piv[k] != k) {
                double t = x[k]; x[k] = x[piv[k]]; x[piv[k]] = t;
            }
            for (i = 0; i < n; i++) for (k = 0; k < i; k++) x[i] -= B[i + k * n] * x[k];
            for (i = n - 1; i >= 0; i--) {
                for (k = i + 1; k < n; k++) x[i] -= B[i + k * n] * x[k];
                x[i] /= B[i + i * n];
            }
        }
    }
    free(B); free(piv);
    return info;
}

#endif
