#include <iostream>
#include <vector>
#include <algorithm>

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
  constexpr int BUFFER_SIZE = 1024;

  int nGPUs;
  CUDA_MUST(cudaGetDeviceCount(&nGPUs));

  if (nGPUs == 0) {
	  std::cerr << "No devices found!\n";
	  exit(EXIT_FAILURE);
  } else {
	std::cout << nGPUs << " devices!" << std::endl;
  }

  // Data that will be on each gpu
  std::vector<std::vector<double>> test_data(   nGPUs, std::vector<double>(BUFFER_SIZE, 1.0));
  std::vector<std::vector<double>> test_results(nGPUs, std::vector<double>(BUFFER_SIZE));
  std::cout << "Allocated host data" << std::endl;

  // associate all devices with corresponding rank
  std::vector<ncclComm_t> comms(nGPUs);
  std::vector<int> devList(nGPUs); 
  std::iota(devList.begin(), devList.end(), 0);
  NCCL_MUST(ncclCommInitAll(&comms[0], nGPUs, &devList[0]));
  std::cout << "Initialized communicators." << std::endl;


  // Create buffers on GPUs
  std::vector<GPUBuffer> GPUBuffers(nGPUs);
  for (auto dev : devList) {
    auto &buf = GPUBuffers[dev];
    CUDA_MUST(cudaSetDevice(dev));
    CUDA_MUST(cudaStreamCreate(&buf.stream_));
    CUDA_MUST(cudaMalloc(&buf.send_, sizeof(double) * BUFFER_SIZE));
    CUDA_MUST(cudaMalloc(&buf.recv_, sizeof(double) * BUFFER_SIZE)); // one value for all-reduce
  }
  std::cout << "Created buffers" << std::endl;

  // Copy test data to GPUs
  for (auto dev : devList) {
    auto &buf = GPUBuffers[dev];
    CUDA_MUST(cudaSetDevice(dev));
    CUDA_MUST(cudaMemcpy(buf.send_, 
			 &test_data[dev][0], 
			 sizeof(double)*test_data[dev].size(), 
			 cudaMemcpyHostToDevice)
	     );
  }
  std::cout << "Copied H2D" << std::endl;

  for (auto i = 0; i < devList.size(); ++i) {
    const auto dev = devList[i];
    CUDA_MUST(cudaSetDevice(dev)); // Correct device must be set prior to each collective call.
    auto &buf = GPUBuffers[dev];
    NCCL_MUST(ncclAllReduce(buf.send_, buf.recv_, BUFFER_SIZE,
        ncclDouble, ncclSum, comms[i], buf.stream_));
  }
  std::cout << "Did allreduce" << std::endl;


  // Check results!
  for (auto dev : devList) {
    auto &buf = GPUBuffers[dev];
    CUDA_MUST(cudaSetDevice(dev));
    CUDA_MUST(cudaMemcpy(&test_results[dev],
			 buf.recv_, 
			 sizeof(double)*test_data[dev].size(), 
			 cudaMemcpyDeviceToHost)
	     );
  }
  std::cout << "Copied D2H" << std::endl;



  for (auto &buf : GPUBuffers) {
	  CUDA_MUST(cudaStreamDestroy(buf.stream_));
	  CUDA_MUST(cudaFree(buf.send_));
	  CUDA_MUST(cudaFree(buf.recv_));
  }
  std::cout << "Cleaned up" << std::endl;
}

