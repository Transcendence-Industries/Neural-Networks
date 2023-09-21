#ifndef IO_H
#define IO_H

#include "network.h"

NNStatus save_network(const char *path, const Network *net);
NNStatus load_network(const char *path, Network *net);

#endif
