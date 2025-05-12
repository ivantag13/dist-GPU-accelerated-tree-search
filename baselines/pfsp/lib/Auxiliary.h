#ifndef AUXILIARY_H
#define AUXILIARY_H

#ifdef __cplusplus
extern "C" {
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

bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag);

void permute(int* arr, int n);

int findMin(int arr[], int size);

int findMaxInt(int arr[], int size);

int compare_doubles(const void *a, const void *b);

double get_min(const double *vec, int D);

double get_max(const double *vec, int D);

double get_median(const double *sorted, int D);

double get_quartile(const double *sorted, int D, double percentile);

double get_stddev(const double *vec, int D);

void compute_boxplot_stats(const double* vec, int D, FILE* file);

#ifdef __cplusplus
}
#endif

#endif // AUXILIARY_H
