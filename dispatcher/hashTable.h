#ifndef HASHTABLE_H
#define HASHTABLE_H
#include "dispatcher.h"

typedef struct{ /* table entry: */
    struct nlist *next; /* next entry in chain */
    char *name; /* defined name */
    dataPacket *defn; /* replacement text */
} nlist;

unsigned hash(char *s);

nlist *lookup(char *s);

char *install(char *name, dataPacket *defn);

char *strdup(char *s); /* make a duplicate of s */

#endif
