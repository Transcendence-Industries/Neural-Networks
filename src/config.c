#include <stdlib.h>
#include "config.h"

unsigned int nn_rng_seed = 1337u;

void nn_set_random_seed(unsigned int seed)
{
    nn_rng_seed = seed;
    srand(seed);
}

TensorType nn_rand_uniform(TensorType min, TensorType max)
{
    TensorType r = (TensorType)rand() / (TensorType)RAND_MAX;
    return min + r * (max - min);
}
