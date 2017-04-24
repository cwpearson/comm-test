#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 3
__global__ void vecAdd(int *a, int *b, int *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

int main() {
        int x = 0;
	int * d_a;
        int * d_b;
        int * d_c;
	int * h_c = (int *)malloc(DSIZE*sizeof(int));
        cudaIpcMemHandle_t a_handle;
        char a_buffer[sizeof(a_handle)+1];
        memset(a_buffer, 0, sizeof(a_handle)+1);
        cudaIpcMemHandle_t b_handle;
        char b_buffer[sizeof(b_handle)+1];
        memset(b_buffer, 0, sizeof(b_handle)+1);
	while(x < 1){
        	FILE * fp;
        	fp = fopen("handlepipe", "r");
        	for (int i = 0; i < sizeof(a_handle); i++){
        	        fscanf(fp,"%c", a_buffer+i);
        	}
        	for (int i = 0; i < sizeof(b_handle); i++){
        	        fscanf(fp,"%c", b_buffer+i);
        	}
		fclose(fp);
        	memcpy((char *)(&a_handle), a_buffer, sizeof(a_handle));
        	cudaIpcOpenMemHandle((void **)&d_a, a_handle, cudaIpcMemLazyEnablePeerAccess);
        	memcpy((char *)(&b_handle), b_buffer, sizeof(b_handle));
        	cudaIpcOpenMemHandle((void **)&d_b, b_handle, cudaIpcMemLazyEnablePeerAccess);
		cudaMalloc(&d_c, DSIZE*sizeof(int));
		int blockSize, gridSize;
    		blockSize = 1024;
     		
    		// Number of thread blocks in grid
    		gridSize = (int)ceil((float)DSIZE/blockSize);    
    		// Execute the kernel
    		vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, DSIZE);
     		
    		// Copy array back to host
    		cudaMemcpy( h_c, d_c, DSIZE*sizeof(int), cudaMemcpyDeviceToHost );
		int sum = 0;
		for(int i = 0; i < DSIZE; i++) {
			sum += h_c[i];
			printf("index %d: %d\n", i, h_c[i]);
		}
		printf("sum %d\n", sum);
		x++;
	}
	return 0;
}
