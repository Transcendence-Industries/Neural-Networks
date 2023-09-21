#include <stdio.h>
#include <string.h>
#include "layers.h"
#include "activation_funcs.h"
#include "config.h"

NNStatus layer_create(Layer *l, int n_inputs, int n_neurons, const ActivationDef *activation)
{
    if (!l || n_inputs <= 0 || n_neurons <= 0 || !activation)
    {
        return NN_ERR_INVALID_ARG;
    }

    memset(l, 0, sizeof(*l));
    l->n_inputs = n_inputs;
    l->n_neurons = n_neurons;
    l->activation = activation;

    NNStatus st = tensor_create(&l->weights, 2, (int[]){n_inputs, n_neurons});
    if (st != NN_OK) return st;
    st = tensor_create(&l->biases, 1, (int[]){n_neurons});
    if (st != NN_OK) { layer_free(l); return st; }
    return NN_OK;
}

void layer_free(Layer *l)
{
    if (!l) return;
    tensor_free(&l->weights);
    tensor_free(&l->biases);
    tensor_free(&l->input_cache);
    tensor_free(&l->z_cache);
    tensor_free(&l->output_cache);
    tensor_free(&l->grad_w);
    tensor_free(&l->grad_b);
}

NNStatus layer_init_random(Layer *l, TensorType min, TensorType max)
{
    if (!l) return NN_ERR_INVALID_ARG;
    for (int i = 0; i < l->weights.n_elements; i++) l->weights.data[i] = nn_rand_uniform(min, max);
    for (int i = 0; i < l->biases.n_elements; i++) l->biases.data[i] = nn_rand_uniform(min, max);
    return NN_OK;
}

void layer_print(const Layer *l)
{
    if (!l) return;
    printf("Layer{in=%d, neurons=%d, activation=%s}\n", l->n_inputs, l->n_neurons, l->activation->name);
}
