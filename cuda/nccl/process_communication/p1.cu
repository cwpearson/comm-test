#include <stdio.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 1

int main()
{
	mkfifo("handlepipe", 0600);
	int *data;
  	int i = 5;
  	cudaMalloc(&data, DSIZE*sizeof(int));
  	cudaMemcpy(data, &i, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
  	cudaIpcMemHandle_t my_handle;
  	cudaIpcGetMemHandle(&my_handle, data);
  	unsigned char handle_buffer[sizeof(my_handle)+1];
  	memset(handle_buffer, 0, sizeof(my_handle)+1);
  	memcpy(handle_buffer, (unsigned char *)(&my_handle), sizeof(my_handle));
  	FILE *fp;
  	fp = fopen("handlepipe", "w");
	for (int i=0; i < sizeof(my_handle); i++) {
		fprintf(fp,"%c", handle_buffer[i]);
	}
	fclose(fp);
	sleep(2);  // wait for p2 to modify data
	int *result = (int *) malloc(DSIZE*sizeof(int));
	cudaMemcpy(result, data, DSIZE*sizeof(int), cudaMemcpyDeviceToHost);
	printf("result: %d\n", *result);
  	
	mkfifo("handlepipe2", 0600);
	fp = fopen("handlepipe2", "w");
	for (int i=0; i < sizeof(my_handle); i++) {
		fprintf(fp,"%c", handle_buffer[i]);
	}
	fclose(fp);
	sleep(2);
	system("rm handlepipe");
	system("rm handlepipe2");
	return 0;
}
