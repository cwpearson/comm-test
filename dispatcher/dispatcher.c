#define SNAME "/wait-sem"
#define SYS_FLAG "/sys_lib_flag"

#include "dispatcher.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <semaphore.h>

static int dispatchExist = 0;
static dataPacket dataArray[100];

int giveToDispatch(int *pointerToObject, int sizeOfObject, int identifier)
{

    sem_t *sem = sem_open(SNAME, O_CREAT, 0644, 0);
    sem_t *sysFlagSem = sem_open(SYS_FLAG, O_CREAT, 0644, 0);
    int semValue;
    sem_getvalue(sysFlagSem, &semValue);
    if (dispatchExist==0)
    {
        // sem_post(sysFlagSem);
        int process_pid = fork();
        if (process_pid == 0)
        {
            launchDispatch();
        }
    }

    const char *name = "/shm-example";// file name
    const int SIZE = 4096;// file size

    int shm_fd;// file descriptor, from shm_open()
    dataPacket *shm_base;// base address, from mmap()

    /* open the shared memory segment as if it was a file */
    shm_fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    ftruncate(shm_fd, 4096);
    if (shm_fd == -1) {
      printf("cons: Shared memory failed: %s\n", strerror(errno));
      return -1;
    }

    /* map the shared memory segment to the address space of the process */
    shm_base = (dataPacket *)mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_base == MAP_FAILED) {
      printf("cons: Map failed: %s\n", strerror(errno));
      // close and unlink?
      return -1;
    }
    dataPacket temporaryPacket;
    temporaryPacket.id = identifier;
    temporaryPacket.pointerToItem = (uint64_t)pointerToObject;
    temporaryPacket.sizeOfItem = sizeOfObject;
    *shm_base = temporaryPacket;

    /* remove the mapped memory segment from the address space of the process */
    if (munmap(shm_base, SIZE) == -1) {
        printf("prod: Unmap failed: %s\n", strerror(errno));
        return -1;
    }

    /* close the shared memory segment as if it was a file */
    if (close(shm_fd) == -1) {
        printf("prod: Close failed: %s\n", strerror(errno));
        return -1;
    }

    // sleep(0.01);
    if (dispatchExist == 0)
    {
        dispatchExist = 1;
        int ret = sem_wait(sem);
    }
    return 0;
}

int* getFromDispatch(int id)
{
    const char *array_name = "/array-access";
    const int arraySize = 100*sizeof(dataPacket);
    int shm_fd_array;
    dataPacket *arrayBase;

    shm_fd_array = shm_open(array_name, O_RDONLY, 0600);
    arrayBase = (dataPacket *)mmap(NULL, 100*sizeof(dataPacket), PROT_READ, MAP_SHARED, shm_fd_array, 0);
    return arrayBase[id].pointerToItem;
}


void launchDispatch()
{
  sem_t *sem = sem_open(SNAME, 0);
  const char *name = "/shm-example";// file name
  const int SIZE = 4096;// file size

  const char *array_name = "/array-access";
  const int arraySize = 100*sizeof(dataPacket);

  int shm_fd_array;
  dataPacket *arrayBase;
  

  shm_fd_array = shm_open(array_name, O_CREAT| O_RDWR, 0600);
  ftruncate(shm_fd_array, 4096);
  if (shm_fd_array == -1)
  {
      printf("Failure to allocate shared array\n");
      exit(1);
  }

  arrayBase = (dataPacket *)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_array, 0);
  if (arrayBase == MAP_FAILED)
    printf("The memory mapping failed!\n");

  /* map the shared memory segment to the address space of the process */
  int shm_fd;// file descriptor, from shm_open()
  shm_fd = shm_open(name, O_RDONLY, 0600);

  sem_post(sem);
  for(;;)
  {
    
    dataPacket *shm_base;// base address, from mmap()
    char *ptr;// shm_base is fixed, ptr is movable
    
      /* create the shared memory segment as if it was a file */
    
    if (shm_fd == -1) {
        printf("prod: Shared memory failed: %s\n", strerror(errno));
        continue;
    }
    shm_base = (dataPacket *)mmap(NULL, SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_base == MAP_FAILED || shm_base[0].id < 0) {
        munmap(shm_base, SIZE);
        continue;
    }
    dataPacket temporaryPacket;
    temporaryPacket = *shm_base;
    arrayBase[temporaryPacket.id] = temporaryPacket;

  }

  exit(1);
}


