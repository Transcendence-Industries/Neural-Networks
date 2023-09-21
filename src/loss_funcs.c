#include <math.h>
#include <string.h>
#include "loss_funcs.h"

static NNStatus check_2d_same_shape(const Tensor *a, const Tensor *b)
{
    if (!a || !b || a->n_dims != 2 || b->n_dims != 2) return NN_ERR_SHAPE;
    if (a->dims[0] != b->dims[0] || a->dims[1] != b->dims[1]) return NN_ERR_SHAPE;
    return NN_OK;
}

static NNStatus mse_value(const Tensor *pred, const Tensor *truth, double *out_loss)
{
    NNStatus st = check_2d_same_shape(pred, truth);
    if (st != NN_OK || !out_loss) return NN_ERR_INVALID_ARG;
    double sum = 0.0;
    for (int i = 0; i < pred->n_elements; i++)
    {
        double d = (double)pred->data[i] - (double)truth->data[i];
        sum += d * d;
    }
    *out_loss = sum / pred->n_elements;
    return NN_OK;
}

static NNStatus mse_grad(Tensor *d_pred, const Tensor *pred, const Tensor *truth)
{
    NNStatus st = check_2d_same_shape(pred, truth);
    if (st != NN_OK) return st;
    if (!d_pred || d_pred->n_elements != pred->n_elements) return NN_ERR_SHAPE;

    double scale = 2.0;
    for (int i = 0; i < pred->n_elements; i++) d_pred->data[i] = (pred->data[i] - truth->data[i]) * scale;
    return NN_OK;
}

static NNStatus bce_value(const Tensor *pred, const Tensor *truth, double *out_loss)
{
    NNStatus st = check_2d_same_shape(pred, truth);
    if (st != NN_OK || !out_loss) return NN_ERR_INVALID_ARG;

    double sum = 0.0;
    for (int i = 0; i < pred->n_elements; i++)
    {
        double p = pred->data[i];
        double y = truth->data[i];
        if (p < NN_EPSILON) p = NN_EPSILON;
        if (p > 1.0 - NN_EPSILON) p = 1.0 - NN_EPSILON;
        sum += y * log(p) + (1.0 - y) * log(1.0 - p);
    }
    *out_loss = -sum / pred->n_elements;
    return NN_OK;
}

static NNStatus bce_grad(Tensor *d_pred, const Tensor *pred, const Tensor *truth)
{
    NNStatus st = check_2d_same_shape(pred, truth);
    if (st != NN_OK || !d_pred || d_pred->n_elements != pred->n_elements) return NN_ERR_SHAPE;

    for (int i = 0; i < pred->n_elements; i++)
    {
        double p = pred->data[i];
        double y = truth->data[i];
        if (p < NN_EPSILON) p = NN_EPSILON;
        if (p > 1.0 - NN_EPSILON) p = 1.0 - NN_EPSILON;
        d_pred->data[i] = (TensorType)((p - y) / (p * (1.0 - p)));
    }
    return NN_OK;
}

static NNStatus cce_value(const Tensor *pred, const Tensor *truth, double *out_loss)
{
    NNStatus st = check_2d_same_shape(pred, truth);
    if (st != NN_OK || !out_loss) return NN_ERR_INVALID_ARG;

    int rows = pred->dims[0], cols = pred->dims[1];
    double sum = 0.0;
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int idx = r * cols + c;
            double p = pred->data[idx];
            if (p < NN_EPSILON) p = NN_EPSILON;
            if (p > 1.0 - NN_EPSILON) p = 1.0 - NN_EPSILON;
            sum += -truth->data[idx] * log(p);
        }
    }
    *out_loss = sum / rows;
    return NN_OK;
}

static NNStatus cce_grad(Tensor *d_pred, const Tensor *pred, const Tensor *truth)
{
    NNStatus st = check_2d_same_shape(pred, truth);
    if (st != NN_OK || !d_pred || d_pred->n_elements != pred->n_elements) return NN_ERR_SHAPE;

    for (int i = 0; i < pred->n_elements; i++) d_pred->data[i] = (pred->data[i] - truth->data[i]);
    return NN_OK;
}

const LossDef LossMSE = {"MSE", mse_value, mse_grad};
const LossDef LossBCE = {"BCE", bce_value, bce_grad};
const LossDef LossCCE = {"CCE", cce_value, cce_grad};

const LossDef *loss_by_name(const char *name)
{
    if (!name) return NULL;
    if (strcmp(name, "MSE") == 0) return &LossMSE;
    if (strcmp(name, "BCE") == 0) return &LossBCE;
    if (strcmp(name, "CCE") == 0) return &LossCCE;
    return NULL;
}
