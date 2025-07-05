#include "util.h"

// Function that performs a swap between two integers
inline void swap(int *a, int *b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

// Function to check if all elements in an array of atomic bool are IDLE
static bool _allIdle(_Atomic bool arr[], int size)
{
  bool value;
  for (int i = 0; i < size; i++)
  {
    value = atomic_load(&arr[i]);
    if (value == BUSY)
    {
      return false;
    }
  }
  return true;
}

// Function to check if all elements in arr are IDLE and update flag accordingly
bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag)
{
  bool value = atomic_load(flag);
  if (value)
  {
    return true; // fast exit
  }
  else
  {
    if (_allIdle(arr, size))
    {
      atomic_store(flag, true);
      return true;
    }
    else
    {
      return false;
    }
  }
}

void permute(int *arr, int n)
{
  for (int i = 0; i < n; i++)
  {
    arr[i] = i;
  }

  // Iterate over each element in the array
  for (int i = n - 1; i > 0; i--)
  {
    // Select a random index from 0 to i (inclusive)
    int j = rand() % (i + 1);

    // Swap arr[i] with the randomly selected element arr[j]
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

// Function to find the minimum value in an array of integers
int findMin(int arr[], int size)
{
  int minVal = arr[0]; // Initialize minVal with the first element

  // Iterate through the array to find the minimum value
  for (int i = 1; i < size; i++)
  {
    if (arr[i] < minVal)
    {
      minVal = arr[i]; // Update minVal if current element is smaller
    }
  }

  return minVal; // Return the minimum value
}

// Function to find the minimum value in an array of integers
int findMaxInt(int arr[], int size)
{
  int maxVal = arr[0]; // Initialize minVal with the first element

  // Iterate through the array to find the minimum value
  for (int i = 1; i < size; i++)
  {
    if (arr[i] > maxVal)
    {
      maxVal = arr[i]; // Update minVal if current element is smaller
    }
  }

  return maxVal; // Return the minimum value
}

int compare_doubles(const void *a, const void *b)
{
  double diff = *(double *)a - *(double *)b;
  return (diff > 0) - (diff < 0);
}

double get_min(const double *vec, int size)
{
  double min = vec[0];
  for (int i = 1; i < size; i++)
    if (vec[i] < min)
      min = vec[i];
  return min;
}

double get_max(const double *vec, int size)
{
  double max = vec[0];
  for (int i = 1; i < size; i++)
    if (vec[i] > max)
      max = vec[i];
  return max;
}

double get_median(const double *sorted, int size)
{
  if (size % 2 == 0)
    return (sorted[size / 2 - 1] + sorted[size / 2]) / 2.0;
  else
    return sorted[size / 2];
}

double get_quartile(const double *sorted, int size, double percentile)
{
  double pos = percentile * (size - 1);
  int lower = (int)pos;
  double delta = pos - lower;
  return sorted[lower] + delta * (sorted[lower + 1] - sorted[lower]);
}

void get_quartiles_tukey(const double *sorted, int size, double *q1, double *q3)
{
  int mid = size / 2;

  if (size % 2 == 0)
  {
    // Even: lower half = [0 .. mid-1], upper half = [mid .. size-1]
    *q1 = get_median(sorted, mid);
    *q3 = get_median(sorted + mid, mid);
  }
  else
  {
    // Odd: lower half = [0 .. mid-1], upper half = [mid+1 .. size-1]
    *q1 = get_median(sorted, mid);
    *q3 = get_median(sorted + mid + 1, mid);
  }
}

double get_percentile(const double *sorted, int n, double pct)
{
  if (n < 1)
    return 0.0;
  double idx = pct * (n - 1);
  int lo = (int)floor(idx);
  double frac = idx - lo;
  if (lo + 1 < n)
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
  else
    return sorted[lo];
}

double get_stddev(const double *vec, int size)
{
  double sum = 0.0, mean, stddev = 0.0;
  for (int i = 0; i < size; i++)
    sum += vec[i];
  mean = sum / size;
  for (int i = 0; i < size; i++)
    stddev += (vec[i] - mean) * (vec[i] - mean);
  return sqrt(stddev / size);
}

void compute_boxplot_stats(const double *vec, int size, FILE *file)
{
  double *sorted = malloc(size * sizeof(double));
  double mean = 0;
  for (int i = 0; i < size; i++)
  {
    sorted[i] = vec[i];
    mean += vec[i];
  }
  mean /= size;
  qsort(sorted, size, sizeof(double), compare_doubles);

  double q1, q3;
  double min = sorted[0];
  double max = sorted[size - 1];
  double median = get_median(sorted, size);
  get_quartiles_tukey(sorted, size, &q1, &q3);
  // double q1 = get_percentile(sorted, size, 0.25);
  // double q3 = get_percentile(sorted, size, 0.75);
  double stddev = get_stddev(vec, size);

  fprintf(file, "Min: %.3f  Q1: %.3f  Median: %.3f  Mean: %.3f  Q3: %.3f  Max: %.3f  StdDev: %.3f\n",
          min, q1, median, mean, q3, max, stddev);

  free(sorted);
}
