#include <iostream>
#include <vector>

#include <nccl.h>

#define CUDA_MUST(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#define NCCL_MUST(ans) \
{ ncclAssert((ans), __FILE__, __LINE__); }
inline void ncclAssert(ncclResult_t result, const char *file, int line,
                      bool abort = true) {
  if (result != ncclSuccess) {
    fprintf(stderr, "nnclAssert: %s %s %d\n", ncclGetErrorString(result), file,
            line);
    if (abort)
      exit(result);
  }
}



typedef struct {
  double* send_;
  double* recv_;
  cudaStream_t stream_;
} GPUBuffer;

__global__ void kernel() {}

int main(int argc, char* argv[]) {
  int nGPUs;
  CUDA_MUST(cudaGetDeviceCount(&nGPUs));

  if (nGPUs == 0) {
	  std::cerr << "No devices found!\n";
	  exit(EXIT_FAILURE);
  }

  // associate all devices with rank 0
  std::vector<ncclComm_t> comms(nGPUs);
  std::vector<int> devList(nGPUs, 0); 
  auto result = ncclCommInitAll(&comms[0], nGPUs, &devList[0]);
  if (result != ncclSuccess) {
    std::cerr << "Error in ncclCommInitAll\n";
  }

  constexpr int BUFFER_SIZE = 1024;

  // Create buffers on GPUs
  std::vector<GPUBuffer> GPUBuffers(nGPUs);
  for (auto dev = 0; dev < GPUBuffers.size(); ++dev) {
	  auto &buf = GPUBuffers[dev];
	  CUDA_MUST(cudaStreamCreate(&buf.stream_));
	  CUDA_MUST(cudaSetDevice(dev));
	  CUDA_MUST(cudaMalloc(&buf.send_, BUFFER_SIZE));
	  CUDA_MUST(cudaMalloc(&buf.recv_, 1)); // for reduce
  }

  // Allocate data and issue work to each GPU's
  // perDevStream to populate the sendBuffs.

  for(int i=0; i<nGPUs; ++i) {
    cudaSetDevice(i); // Correct device must be set
                      // prior to each collective call.
    auto &buf = GPUBuffers[i];
    ncclAllReduce(buf.send_, buf.recv_, BUFFER_SIZE,
        ncclDouble, ncclSum, comms[i], buf.stream_);
  }

  // Issue work into data[*].stream to consume buffers, etc.


  for (auto &buf : GPUBuffers) {
	  CUDA_MUST(cudaStreamDestroy(buf.stream_));
	  CUDA_MUST(cudaFree(buf.send_));
	  CUDA_MUST(cudaFree(buf.recv_));
  }
}

