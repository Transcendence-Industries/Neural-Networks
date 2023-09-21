#include <math.h>
#include <string.h>
#include "activation_funcs.h"

static NNStatus activation_identity_forward(Tensor *out, const Tensor *in)
{
    if (!out || !in) return NN_ERR_INVALID_ARG;
    return tensor_copy(out, in);
}

static NNStatus activation_identity_backward(Tensor *d_out, const Tensor *activated_out)
{
    (void)activated_out;
    if (!d_out) return NN_ERR_INVALID_ARG;
    return NN_OK;
}

static NNStatus activation_relu_forward(Tensor *out, const Tensor *in)
{
    NNStatus st = tensor_copy(out, in);
    if (st != NN_OK) return st;
    for (int i = 0; i < out->n_elements; i++) if (out->data[i] < 0.0) out->data[i] = 0.0;
    return NN_OK;
}

static NNStatus activation_relu_backward(Tensor *d_out, const Tensor *activated_out)
{
    if (!d_out || !activated_out || d_out->n_elements != activated_out->n_elements) return NN_ERR_INVALID_ARG;
    for (int i = 0; i < d_out->n_elements; i++) d_out->data[i] *= (activated_out->data[i] > 0.0) ? 1.0 : 0.0;
    return NN_OK;
}

static NNStatus activation_sigmoid_forward(Tensor *out, const Tensor *in)
{
    NNStatus st = tensor_copy(out, in);
    if (st != NN_OK) return st;
    for (int i = 0; i < out->n_elements; i++) out->data[i] = 1.0 / (1.0 + exp(-out->data[i]));
    return NN_OK;
}

static NNStatus activation_sigmoid_backward(Tensor *d_out, const Tensor *activated_out)
{
    if (!d_out || !activated_out || d_out->n_elements != activated_out->n_elements) return NN_ERR_INVALID_ARG;
    for (int i = 0; i < d_out->n_elements; i++)
    {
        TensorType y = activated_out->data[i];
        d_out->data[i] *= y * (1.0 - y);
    }
    return NN_OK;
}

static NNStatus activation_tanh_forward(Tensor *out, const Tensor *in)
{
    NNStatus st = tensor_copy(out, in);
    if (st != NN_OK) return st;
    for (int i = 0; i < out->n_elements; i++) out->data[i] = tanh(out->data[i]);
    return NN_OK;
}

static NNStatus activation_tanh_backward(Tensor *d_out, const Tensor *activated_out)
{
    if (!d_out || !activated_out || d_out->n_elements != activated_out->n_elements) return NN_ERR_INVALID_ARG;
    for (int i = 0; i < d_out->n_elements; i++)
    {
        TensorType y = activated_out->data[i];
        d_out->data[i] *= (1.0 - y * y);
    }
    return NN_OK;
}

static NNStatus activation_softmax_forward(Tensor *out, const Tensor *in)
{
    NNStatus st = tensor_copy(out, in);
    if (st != NN_OK) return st;
    if (out->n_dims != 2) return NN_ERR_SHAPE;

    int rows = out->dims[0];
    int cols = out->dims[1];
    for (int r = 0; r < rows; r++)
    {
        int off = r * cols;
        TensorType max_v = out->data[off];
        for (int c = 1; c < cols; c++) if (out->data[off + c] > max_v) max_v = out->data[off + c];

        TensorType sum = 0.0;
        for (int c = 0; c < cols; c++)
        {
            out->data[off + c] = exp(out->data[off + c] - max_v);
            sum += out->data[off + c];
        }
        for (int c = 0; c < cols; c++) out->data[off + c] /= sum;
    }
    return NN_OK;
}

static NNStatus activation_softmax_backward(Tensor *d_out, const Tensor *activated_out)
{
    (void)activated_out;
    if (!d_out) return NN_ERR_INVALID_ARG;
    return NN_OK;
}

const ActivationDef ActivationIdentity = {"Identity", activation_identity_forward, activation_identity_backward};
const ActivationDef ActivationRelu = {"ReLU", activation_relu_forward, activation_relu_backward};
const ActivationDef ActivationSigmoid = {"Sigmoid", activation_sigmoid_forward, activation_sigmoid_backward};
const ActivationDef ActivationTanh = {"Tanh", activation_tanh_forward, activation_tanh_backward};
const ActivationDef ActivationSoftmax = {"Softmax", activation_softmax_forward, activation_softmax_backward};

const ActivationDef *activation_by_name(const char *name)
{
    if (!name) return NULL;
    if (strcmp(name, "Identity") == 0) return &ActivationIdentity;
    if (strcmp(name, "ReLU") == 0) return &ActivationRelu;
    if (strcmp(name, "Sigmoid") == 0) return &ActivationSigmoid;
    if (strcmp(name, "Tanh") == 0) return &ActivationTanh;
    if (strcmp(name, "Softmax") == 0) return &ActivationSoftmax;
    return NULL;
}
