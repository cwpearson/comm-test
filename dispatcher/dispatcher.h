#ifndef DISPATCHER_H
#define DISPATCHER_H
#include <stdint.h>

typedef struct{
    int id;
    uint64_t pointerToItem;
    int sizeOfItem;
} dataPacket;

int giveToDispatch(int * pointerToObject, int sizeOfObject, int identifier);

void launchDispatch();


#endif
