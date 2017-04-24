#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 3

int main()
{
        mkfifo("handlepipe", 0600);
	int x = 0;
        while(true) {
		int * h_a = (int*) malloc(DSIZE*sizeof(int));
        	int * h_b = (int*) malloc(DSIZE*sizeof(int));
        	int * h_c = (int*) malloc(DSIZE*sizeof(int));
        	for(int i = 0; i < DSIZE; i++) {
        	        h_a[i] = x+i;
        	        h_b[i] = 3;
        	}
        	int * d_a;
        	int * d_b;
		
        	cudaMalloc(&d_a, DSIZE*sizeof(int));
        	cudaMemcpy(d_a, h_a, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
        	cudaIpcMemHandle_t a_handle;
       		cudaIpcGetMemHandle(&a_handle, d_a);
        	unsigned char a_buffer[sizeof(a_handle)+1];
        	memset(a_buffer, 0, sizeof(a_handle)+1);
        	memcpy(a_buffer, (unsigned char *)(&a_handle), sizeof(a_handle));
	
        	cudaMalloc(&d_b, DSIZE*sizeof(int));
        	cudaMemcpy(d_b, h_b, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
        	cudaIpcMemHandle_t b_handle;
        	cudaIpcGetMemHandle(&b_handle, d_b);
        	unsigned char b_buffer[sizeof(b_handle)+1];
        	memset(b_buffer, 0, sizeof(b_handle)+1);
        	memcpy(b_buffer, (unsigned char *)(&b_handle), sizeof(b_handle));
	
        	FILE * fp;
        	printf("waiting for other side to open\n");
        	fp = fopen("handlepipe", "w");
        	for (int i=0; i < sizeof(a_handle); i++) {
                	fprintf(fp,"%c", a_buffer[i]);
        	}
        	for (int i=0; i < sizeof(b_handle); i++) {
                	fprintf(fp,"%c", b_buffer[i]);
        	}
        	fclose(fp);
		x++;
        }
	sleep(2);  // wait for p2 to modify dat
        system("rm handlepipe");
        return 0;
}
