#include <nccl.h>

typedef struct {
  double* sendBuff;
  double* recvBuff;
  int size;
  cudaStream_t stream;
} PerThreadData;

int main(int argc, char* argv[])
{
  int nGPUs;
  cudaGetDeviceCount(&nGPUs);
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nGPUs);
  ncclCommInitAll(comms, nGPUs); // initialize communicator
                                // One communicator per process

  PerThreadData* data;

  ... // Allocate data and issue work to each GPU's
      // perDevStream to populate the sendBuffs.

  for(int i=0; i<nGPUs; ++i) {
    cudaSetDevice(i); // Correct device must be set
                      // prior to each collective call.
    ncclAllReduce(data[i].sendBuff, data[i].recvBuff, size,
        ncclDouble, ncclSum, comms[i], data[i].stream);
  }

  ... // Issue work into data[*].stream to consume buffers, etc.
}
