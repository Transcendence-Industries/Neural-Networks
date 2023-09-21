#ifndef NETWORK_H
#define NETWORK_H

#include "calc.h"
#include "loss_funcs.h"
#include "accuracy_funcs.h"

typedef struct
{
    Layer *layers;
    int n_layers;
    int capacity;
} Network;

typedef struct
{
    int epochs;
    int batch_size;
    TensorType learning_rate;
    const LossDef *loss;
    const AccuracyDef *accuracy;
    int verbose;
} FitConfig;

NNStatus network_create(Network *net, int capacity);
void network_free(Network *net);
NNStatus network_add_layer(Network *net, int n_inputs, int n_neurons, const ActivationDef *activation, TensorType init_min, TensorType init_max);
NNStatus network_predict(Network *net, const Tensor *inputs, Tensor *outputs);
NNStatus network_evaluate(Network *net, const Tensor *inputs, const Tensor *truth, const LossDef *loss, const AccuracyDef *accuracy, double *out_loss, double *out_acc);
NNStatus network_fit(Network *net, const Tensor *x_train, const Tensor *y_train, const FitConfig *cfg);

#endif
