#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --account=project_465002032
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00

# === Default values ===
MIN_SIZE=25
MAX_SIZE=50000
REPETITIONS=3
JOBS=0
MACHINES=0
LEVEL=1

# === Parse command-line options ===
while getopts ":m:M:r:j:g:l:" opt; do
  case $opt in
    m) MIN_SIZE=$OPTARG ;;
    M) MAX_SIZE=$OPTARG ;;
    p) REPETITIONS=$OPTARG ;;
    j) JOBS=$OPTARG ;;
    g) MACHINES=$OPTARG ;;
    l) LEVEL=$OPTARG ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# === Define known Taillard instance groups ===
INSTANCES=""

if [ "$JOBS" -eq 20 ]; then
  if [ "$MACHINES" -eq 5 ]; then
    INSTANCES="1 2 3 4 5 6 7 8 9 10"
  elif [ "$MACHINES" -eq 10 ]; then
    INSTANCES="11 12 13 14 15 16 17 18 19 20"
  elif [ "$MACHINES" -eq 20 ]; then
    INSTANCES="29 30 22 27 23 28 25 26 24 21"
  fi
elif [ "$JOBS" -eq 50 ]; then
  if [ "$MACHINES" -eq 5 ]; then
    INSTANCES="31 32 33 34 35 36 37 38 39 40"
  elif [ "$MACHINES" -eq 10 ]; then
    INSTANCES="41 42 43 44 45 46 47 48 49 50"
  elif [ "$MACHINES" -eq 20 ]; then
    INSTANCES="52 53 56 57 58"
  fi
elif [ "$JOBS" -eq 100 ]; then
  if [ "$MACHINES" -eq 5 ]; then
    INSTANCES="61 62 63 64 65 66 67 68 69 70"
  elif [ "$MACHINES" -eq 10 ]; then
    INSTANCES="71 72 73 74 75 76 77 78 79 80"
  elif [ "$MACHINES" -eq 20 ]; then
    INSTANCES="82 82 84 90"
  fi
elif [ "$JOBS" -eq 200 ]; then
  if [ "$MACHINES" -eq 10 ]; then
    INSTANCES="91 92 93 94 95 96 97 98 99 100"
  elif [ "$MACHINES" -eq 20 ]; then
    INSTANCES="101 103 104 105 106 107 108 109 110"
  fi
elif [ "$JOBS" -eq 500 ] && [ "$MACHINES" -eq 20 ]; then
  INSTANCES="111 112 113 114 115 116 117 118 119 120"
fi

if [ -z "$INSTANCES" ]; then
  echo "Error: No instance group defined for j=$JOBS and g=$MACHINES" >&2
  exit 1
fi


echo "Running instances for jobs=$JOBS and machines=$MACHINES"
echo "Instances: $INSTANCES"
echo "Repetitions per instance: $REPETITIONS"
echo "Min size (m): $MIN_SIZE, Max size (M): $MAX_SIZE"
echo "------------------------------------------"

for k in $INSTANCES; do
  for ((i=1; i<=REPETITIONS; i++)); do
    echo "Running instance $k (rep $i)"
    srun ./../pfsp_gpu_hip.out -i "$k" -l "$LEVEL" -m "$MIN_SIZE" -M "$MAX_SIZE"
  done
done

# Still unresolved Taillard instances are left out, to know: ta051, ta054, ta055, ta059, ta060, ta081, ta085, ta086. ta087. ta088, ta089, ta102