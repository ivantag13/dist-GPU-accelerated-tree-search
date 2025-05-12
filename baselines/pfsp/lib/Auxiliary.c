#include "Auxiliary.h"

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

double get_min(const double *vec, int D)
{
  double min = vec[0];
  for (int i = 1; i < D; i++)
    if (vec[i] < min)
      min = vec[i];
  return min;
}

double get_max(const double *vec, int D)
{
  double max = vec[0];
  for (int i = 1; i < D; i++)
    if (vec[i] > max)
      max = vec[i];
  return max;
}

double get_median(const double *sorted, int D)
{
  if (D % 2 == 0)
    return (sorted[D / 2 - 1] + sorted[D / 2]) / 2.0;
  else
    return sorted[D / 2];
}

double get_quartile(const double *sorted, int D, double percentile)
{
  double pos = percentile * (D - 1);
  int lower = (int)pos;
  double delta = pos - lower;
  return sorted[lower] + delta * (sorted[lower + 1] - sorted[lower]);
}

double get_stddev(const double *vec, int D)
{
  double sum = 0.0, mean, stddev = 0.0;
  for (int i = 0; i < D; i++)
    sum += vec[i];
  mean = sum / D;
  for (int i = 0; i < D; i++)
    stddev += (vec[i] - mean) * (vec[i] - mean);
  return sqrt(stddev / D);
}

void compute_boxplot_stats(const double *vec, int D, FILE *file)
{
  double *sorted = malloc(D * sizeof(double));
  for (int i = 0; i < D; i++)
    sorted[i] = vec[i];
  qsort(sorted, D, sizeof(double), compare_doubles);

  double min = sorted[0];
  double max = sorted[D - 1];
  double q1 = get_quartile(sorted, D, 0.25);
  double median = get_median(sorted, D);
  double q3 = get_quartile(sorted, D, 0.75);
  double stddev = get_stddev(vec, D);

  fprintf(file, "Min: %.3f  Q1: %.3f  Median: %.3f  Q3: %.3f  Max: %.3f  StdDev: %.3f\n",
          min, q1, median, q3, max, stddev);

  free(sorted);
}
