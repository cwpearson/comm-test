#! /bin/bash

set -eou pipefail

if [[ ! -f $1 ]]; then
  echo "Expected a single executable as an argument"
  exit
else
  echo "Profiling" $1
fi

NVPROF_COMMON_ARGS="-f --unified-memory-profiling per-process-device"

nvprof $NVPROF_COMMON_ARGS -o timeline.nvprof $1
nvprof $NVPROF_COMMON_ARGS -o analysis.nvprof --analysis-metrics $1