#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include "activation_funcs.h"

typedef struct
{
    int n_inputs;
    int n_neurons;
    Tensor weights;
    Tensor biases;
    const ActivationDef *activation;

    Tensor input_cache;
    Tensor z_cache;
    Tensor output_cache;
    Tensor grad_w;
    Tensor grad_b;
} Layer;

NNStatus layer_create(Layer *l, int n_inputs, int n_neurons, const ActivationDef *activation);
void layer_free(Layer *l);
NNStatus layer_init_random(Layer *l, TensorType min, TensorType max);
void layer_print(const Layer *l);

#endif
