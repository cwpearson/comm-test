#include <stdio.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int main(int argc, char const **argv)
{
	int nStreams = 4;
	int blockSize = 256;
	int n = 4 * 1024 * blockSize * nStreams;
	int streamSize = n / nStreams;
	int streamBytes = streamSize * sizeof(float);
	int bytes = n * sizeof(float);

	int devId = 0;
	if (argc > 1) devId = atoi(argv[1]);

	cudaDeviceProp prop;
	checkCuda( cudaGetDeviceProperties(&prop, devId));
	printf("Device : %s\n", prop.name);
	checkCuda( cudaSetDevice(devId) );

	// allocate pinned host memory and device memory
	// non unified
	float *a, *d_a;
	checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
	checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

	float ms; // elapsed time in milliseconds

	//create streams
	cudaStream_t stream[nStreams];

	//events for profiling
	cudaEvent_t startEvent, stopEvent; 

	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );
	for (int i = 0; i < nStreams; ++i) checkCuda( cudaStreamCreate(&stream[i]) );

	//sequential memory transfer 
	memset(a, 0, bytes);
	checkCuda( cudaEventRecord(startEvent) );
	checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
	checkCuda( cudaEventRecord(stopEvent) );
	checkCuda( cudaEventSynchronize(stopEvent));
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf(" Sequential Memory Test\n");
	printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / ms);
	printf(" Host to device tranfser time (ms): %f\n", ms);
	
	checkCuda( cudaEventRecord(startEvent));
	checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
	checkCuda( cudaEventRecord(stopEvent));
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / ms);
	printf(" Device to host tranfser time (ms): %f\n", ms);

	//asynchronous version: loop  over streams{copy H2D, copy D2H}
	memset(a, 0, bytes);
	checkCuda( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
		                           streamBytes, cudaMemcpyHostToDevice, 
		                           stream[i]) );
	}
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf( "Asynchronous Memory Test\n");
	printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / ms);
	printf(" Host to device tranfser time (ms): %f\n", ms);

	checkCuda( cudaEventRecord(startEvent,0) );
	for (int i = 0; i < nStreams; ++i) {
		int offset = i*streamSize;
		checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
		                           streamBytes, cudaMemcpyDeviceToHost,
		                           stream[i]) );
	}
	checkCuda( cudaEventRecord(stopEvent));
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / ms);
	printf(" Device to host tranfser time (ms): %f\n", ms);


	// cleanup
	checkCuda( cudaEventDestroy(startEvent) );
	checkCuda( cudaEventDestroy(stopEvent) );
	for (int i = 0; i < nStreams; ++i)
		checkCuda( cudaStreamDestroy(stream[i]) );
	cudaFree(d_a);
	cudaFreeHost(a);

	return 0;
}
