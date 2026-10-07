/* Batch entry point for RTKLIB lambda(): many independent ILS problems in one
 * call, so Python pays the call overhead once. Problems are packed back to
 * back: a[off_a[i] .. off_a[i]+n[i]), Q[off_q[i] .. off_q[i]+n[i]^2)
 * (column-major), F[2*off_a[i] ..) receives the best and second-best integer
 * vectors, s[2*i ..) their squared residuals, info[i] the status. */
#include "lambda.c"

extern void lambda_batch(int nprob, const int *n, const long *off_a,
                         const long *off_q, const double *a, const double *Q,
                         double *F, double *s, int *info)
{
    int i;
#pragma omp parallel for schedule(dynamic, 64)
    for (i = 0; i < nprob; i++) {
        info[i] = lambda(n[i], 2, a + off_a[i], Q + off_q[i], F + 2 * off_a[i],
                         s + 2 * i);
    }
}
