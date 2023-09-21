#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

typedef double TensorType;

typedef enum
{
    NN_OK = 0,
    NN_ERR_INVALID_ARG = 1,
    NN_ERR_SHAPE = 2,
    NN_ERR_ALLOC = 3,
    NN_ERR_IO = 4,
    NN_ERR_FORMAT = 5
} NNStatus;

#define NN_EPSILON 1e-12

extern unsigned int nn_rng_seed;

void nn_set_random_seed(unsigned int seed);
TensorType nn_rand_uniform(TensorType min, TensorType max);

#endif
