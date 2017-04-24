#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 1
int main(){
	int * data;
	cudaIpcMemHandle_t data_handle;
	char handle_buffer[sizeof(data_handle)+1];
	memset(handle_buffer, 0, sizeof(data_handle)+1);
	FILE * fp;
	fp = fopen("handlepipe2", "r");
	for (int i = 0; i < sizeof(data_handle); i++){
		fscanf(fp,"%c", handle_buffer+i);
	}
	memcpy((char *)(&data_handle), handle_buffer, sizeof(data_handle));
	cudaIpcOpenMemHandle((void **)&data, data_handle, cudaIpcMemLazyEnablePeerAccess);
	int * result = (int *) malloc(DSIZE*sizeof(int));
	cudaMemcpy(result, data, sizeof(int), cudaMemcpyDeviceToHost);
	printf("result: %d\n", *result);
	return 0;
}
