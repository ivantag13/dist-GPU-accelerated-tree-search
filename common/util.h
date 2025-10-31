#ifndef UTIL_H
#define UTIL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define BUSY false
#define IDLE true

    /******************************************************************************
    Auxiliary functions
    ******************************************************************************/

    static inline void swap(__int16_t *a, __int16_t *b)
    {
        __int16_t tmp = *b;
        *b = *a;
        *a = tmp;
    }

    bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag);

    void permute(int *arr, int n);

    int findMin(int arr[], int size);

    int findMaxInt(int arr[], int size);

    int compare_doubles(const void *a, const void *b);

    double get_min(const double *vec, int size);

    double get_max(const double *vec, int size);

    double get_median(const double *sorted, int size);

    double get_quartile(const double *sorted, int size, double percentile);

    void get_quartiles_tukey(const double *sorted, int size, double *q1, double *q3);

    double get_percentile(const double *sorted, int n, double pct);

    double get_stddev(const double *vec, int size);

    void compute_boxplot_stats(const double *vec, int size, FILE *file);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
