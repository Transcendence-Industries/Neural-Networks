#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "network.h"

NNStatus network_create(Network *net, int capacity)
{
    if (!net || capacity <= 0) return NN_ERR_INVALID_ARG;
    net->layers = (Layer *)calloc((size_t)capacity, sizeof(Layer));
    if (!net->layers) return NN_ERR_ALLOC;
    net->n_layers = 0;
    net->capacity = capacity;
    return NN_OK;
}

void network_free(Network *net)
{
    if (!net) return;
    for (int i = 0; i < net->n_layers; i++) layer_free(&net->layers[i]);
    free(net->layers);
    net->layers = NULL;
    net->n_layers = 0;
    net->capacity = 0;
}

NNStatus network_add_layer(Network *net, int n_inputs, int n_neurons, const ActivationDef *activation, TensorType init_min, TensorType init_max)
{
    if (!net || !activation || net->n_layers >= net->capacity) return NN_ERR_INVALID_ARG;
    if (net->n_layers > 0)
    {
        int prev = net->layers[net->n_layers - 1].n_neurons;
        if (prev != n_inputs) return NN_ERR_SHAPE;
    }

    NNStatus st = layer_create(&net->layers[net->n_layers], n_inputs, n_neurons, activation);
    if (st != NN_OK) return st;
    st = layer_init_random(&net->layers[net->n_layers], init_min, init_max);
    if (st != NN_OK) return st;
    net->n_layers++;
    return NN_OK;
}

NNStatus network_predict(Network *net, const Tensor *inputs, Tensor *outputs)
{
    if (!net || !inputs || !outputs || net->n_layers <= 0) return NN_ERR_INVALID_ARG;

    Tensor current = {0};
    NNStatus st = tensor_clone(&current, inputs);
    if (st != NN_OK) return st;

    // Run a full forward pass layer-by-layer, carrying the latest activation tensor.
    for (int i = 0; i < net->n_layers; i++)
    {
        Tensor next = {0};
        st = layer_forward(&net->layers[i], &current, &next);
        tensor_free(&current);
        if (st != NN_OK)
        {
            tensor_free(&next);
            return st;
        }
        current = next;
    }

    tensor_free(outputs);
    *outputs = current;
    return NN_OK;
}

NNStatus network_evaluate(Network *net, const Tensor *inputs, const Tensor *truth, const LossDef *loss, const AccuracyDef *accuracy, double *out_loss, double *out_acc)
{
    if (!loss || !accuracy || !out_loss || !out_acc) return NN_ERR_INVALID_ARG;

    Tensor pred = {0};
    NNStatus st = network_predict(net, inputs, &pred);
    if (st != NN_OK) return st;

    st = loss->value(&pred, truth, out_loss);
    if (st == NN_OK) st = accuracy->value(&pred, truth, out_acc);
    tensor_free(&pred);
    return st;
}

NNStatus network_fit(Network *net, const Tensor *x_train, const Tensor *y_train, const FitConfig *cfg)
{
    if (!net || !x_train || !y_train || !cfg || !cfg->loss || !cfg->accuracy) return NN_ERR_INVALID_ARG;
    if (x_train->n_dims != 2 || y_train->n_dims != 2 || x_train->dims[0] != y_train->dims[0]) return NN_ERR_SHAPE;

    int n_samples = x_train->dims[0];
    int input_dim = x_train->dims[1];
    int out_dim = y_train->dims[1];
    if (net->layers[0].n_inputs != input_dim || net->layers[net->n_layers - 1].n_neurons != out_dim) return NN_ERR_SHAPE;

    int batch_size = cfg->batch_size > 0 ? cfg->batch_size : n_samples;

    for (int epoch = 0; epoch < cfg->epochs; epoch++)
    {
        // Mini-batch SGD over the full training set.
        for (int start = 0; start < n_samples; start += batch_size)
        {
            int end = start + batch_size;
            if (end > n_samples) end = n_samples;
            int bs = end - start;

            Tensor batch_x = {0}, batch_y = {0};
            NNStatus st = tensor_create(&batch_x, 2, (int[]){bs, input_dim});
            if (st != NN_OK) return st;
            st = tensor_create(&batch_y, 2, (int[]){bs, out_dim});
            if (st != NN_OK) { tensor_free(&batch_x); return st; }

            memcpy(batch_x.data, x_train->data + (start * input_dim), (size_t)bs * input_dim * sizeof(TensorType));
            memcpy(batch_y.data, y_train->data + (start * out_dim), (size_t)bs * out_dim * sizeof(TensorType));

            Tensor pred = {0};
            st = network_predict(net, &batch_x, &pred);
            if (st != NN_OK) { tensor_free(&batch_x); tensor_free(&batch_y); return st; }

            Tensor grad = {0};
            st = tensor_create(&grad, 2, (int[]){bs, out_dim});
            if (st != NN_OK) { tensor_free(&batch_x); tensor_free(&batch_y); tensor_free(&pred); return st; }
            st = cfg->loss->grad(&grad, &pred, &batch_y);
            if (st != NN_OK) { tensor_free(&batch_x); tensor_free(&batch_y); tensor_free(&pred); tensor_free(&grad); return st; }

            // Backpropagate loss gradient from output layer to input layer.
            Tensor upstream = grad;
            for (int i = net->n_layers - 1; i >= 0; i--)
            {
                Tensor next = {0};
                st = layer_backward(&net->layers[i], &upstream, &next);
                tensor_free(&upstream);
                if (st != NN_OK)
                {
                    tensor_free(&next);
                    tensor_free(&batch_x); tensor_free(&batch_y); tensor_free(&pred);
                    return st;
                }
                // Update parameters using gradients accumulated across this batch.
                st = layer_apply_sgd(&net->layers[i], cfg->learning_rate, bs);
                if (st != NN_OK)
                {
                    tensor_free(&next);
                    tensor_free(&batch_x); tensor_free(&batch_y); tensor_free(&pred);
                    return st;
                }
                upstream = next;
            }

            tensor_free(&upstream);
            tensor_free(&pred);
            tensor_free(&batch_x);
            tensor_free(&batch_y);
        }

        if (cfg->verbose)
        {
            double loss_v = 0.0, acc_v = 0.0;
            NNStatus est = network_evaluate(net, x_train, y_train, cfg->loss, cfg->accuracy, &loss_v, &acc_v);
            if (est == NN_OK)
            {
                printf("Epoch %d/%d loss=%.6f acc=%.6f\n", epoch + 1, cfg->epochs, loss_v, acc_v);
            }
        }
    }

    return NN_OK;
}
