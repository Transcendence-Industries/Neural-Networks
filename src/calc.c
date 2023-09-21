#include <string.h>
#include "calc.h"

static NNStatus ensure_cache_shape(Tensor *t, int n_dims, const int *dims)
{
    // Reuse existing storage when possible to reduce allocations during training.
    if (t->data)
    {
        if (t->n_dims == n_dims)
        {
            int same = 1;
            for (int i = 0; i < n_dims; i++) if (t->dims[i] != dims[i]) same = 0;
            if (same) return NN_OK;
        }
        tensor_free(t);
    }
    return tensor_create(t, n_dims, dims);
}

NNStatus layer_forward(Layer *l, const Tensor *inputs, Tensor *outputs)
{
    if (!l || !inputs || !outputs || inputs->n_dims != 2 || inputs->dims[1] != l->n_inputs)
    {
        return NN_ERR_SHAPE;
    }

    int batch = inputs->dims[0];
    NNStatus st = ensure_cache_shape(&l->input_cache, 2, (int[]){batch, l->n_inputs});
    if (st != NN_OK) return st;
    st = tensor_copy(&l->input_cache, inputs);
    if (st != NN_OK) return st;

    st = ensure_cache_shape(&l->z_cache, 2, (int[]){batch, l->n_neurons});
    if (st != NN_OK) return st;

    for (int r = 0; r < batch; r++)
    {
        for (int c = 0; c < l->n_neurons; c++)
        {
            double sum = l->biases.data[c];
            for (int k = 0; k < l->n_inputs; k++)
            {
                sum += inputs->data[r * l->n_inputs + k] * l->weights.data[k * l->n_neurons + c];
            }
            l->z_cache.data[r * l->n_neurons + c] = (TensorType)sum;
        }
    }

    st = ensure_cache_shape(outputs, 2, (int[]){batch, l->n_neurons});
    if (st != NN_OK) return st;
    st = l->activation->forward(outputs, &l->z_cache);
    if (st != NN_OK) return st;

    st = ensure_cache_shape(&l->output_cache, 2, (int[]){batch, l->n_neurons});
    if (st != NN_OK) return st;
    return tensor_copy(&l->output_cache, outputs);
}

NNStatus layer_backward(Layer *l, const Tensor *d_out, Tensor *d_inputs)
{
    if (!l || !d_out || !d_inputs || d_out->n_dims != 2 || d_out->dims[1] != l->n_neurons)
    {
        return NN_ERR_SHAPE;
    }
    int batch = d_out->dims[0];
    if (l->input_cache.n_dims != 2 || l->input_cache.dims[0] != batch) return NN_ERR_SHAPE;

    Tensor d_z = {0};
    NNStatus st = tensor_clone(&d_z, d_out);
    if (st != NN_OK) return st;

    // For softmax + cross-entropy, the incoming gradient is already dL/dz.
    if (l->activation != &ActivationSoftmax)
    {
        st = l->activation->backward(&d_z, &l->output_cache);
        if (st != NN_OK) { tensor_free(&d_z); return st; }
    }

    st = ensure_cache_shape(&l->grad_w, 2, (int[]){l->n_inputs, l->n_neurons});
    if (st != NN_OK) { tensor_free(&d_z); return st; }
    st = tensor_fill(&l->grad_w, 0.0);
    if (st != NN_OK) { tensor_free(&d_z); return st; }

    st = ensure_cache_shape(&l->grad_b, 1, (int[]){l->n_neurons});
    if (st != NN_OK) { tensor_free(&d_z); return st; }
    st = tensor_fill(&l->grad_b, 0.0);
    if (st != NN_OK) { tensor_free(&d_z); return st; }

    st = ensure_cache_shape(d_inputs, 2, (int[]){batch, l->n_inputs});
    if (st != NN_OK) { tensor_free(&d_z); return st; }
    st = tensor_fill(d_inputs, 0.0);
    if (st != NN_OK) { tensor_free(&d_z); return st; }

    for (int r = 0; r < batch; r++)
    {
        for (int c = 0; c < l->n_neurons; c++)
        {
            TensorType dz = d_z.data[r * l->n_neurons + c];
            l->grad_b.data[c] += dz;
            for (int k = 0; k < l->n_inputs; k++)
            {
                l->grad_w.data[k * l->n_neurons + c] += l->input_cache.data[r * l->n_inputs + k] * dz;
                d_inputs->data[r * l->n_inputs + k] += l->weights.data[k * l->n_neurons + c] * dz;
            }
        }
    }

    tensor_free(&d_z);
    return NN_OK;
}

NNStatus layer_apply_sgd(Layer *l, TensorType lr, int batch_size)
{
    if (!l || batch_size <= 0 || !l->grad_w.data || !l->grad_b.data) return NN_ERR_INVALID_ARG;
    TensorType scale = lr / batch_size;
    for (int i = 0; i < l->weights.n_elements; i++) l->weights.data[i] -= scale * l->grad_w.data[i];
    for (int i = 0; i < l->biases.n_elements; i++) l->biases.data[i] -= scale * l->grad_b.data[i];
    return NN_OK;
}
