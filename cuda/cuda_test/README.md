Code to get CUDA device properties and launch a kernel.
Checks that CUDA and GPUs are set up properly

    cd build
    cmake ..
    make
    ./cuda-test

## Profiling

    cd build
    cmake ..
    make
    ./profile.sh ./cuda-test

You may see a message like `unified memory profiling failed`. This may be
resolved by running `nvprof` with sudo (either by running the profile script
with sudo, or by modifying the profile script).

The script profiles the executable twice. First, to generate a `timeline.nvprof`
file, and second to generate an `analysis.nvprof` file. Together, those files
may be viewed in `nvvp`.

## Feature Wishlist

* ~~Build with NVCC~~
* ~~Device query~~
* ~~Kernel launch~~
* ~~Multiple devices~~
* Build with Clang
* ~~nvprof example~~
* cudnn
* cublas 
