#include <math.h>
#include "accuracy_funcs.h"

static NNStatus accuracy_regression(const Tensor *pred, const Tensor *truth, double *out_acc)
{
    if (!pred || !truth || !out_acc || pred->n_elements != truth->n_elements) return NN_ERR_INVALID_ARG;
    int ok = 0;
    for (int i = 0; i < pred->n_elements; i++) if (fabs(pred->data[i] - truth->data[i]) < 0.08) ok++;
    *out_acc = (double)ok / pred->n_elements;
    return NN_OK;
}

static NNStatus accuracy_categorical(const Tensor *pred, const Tensor *truth, double *out_acc)
{
    if (!pred || !truth || !out_acc || pred->n_dims != 2 || truth->n_dims != 2) return NN_ERR_INVALID_ARG;
    if (pred->dims[0] != truth->dims[0] || pred->dims[1] != truth->dims[1]) return NN_ERR_SHAPE;

    int rows = pred->dims[0];
    int ok = 0;
    for (int r = 0; r < rows; r++)
    {
        int pi = tensor_argmax_row(pred, r);
        int ti = tensor_argmax_row(truth, r);
        if (pi == ti) ok++;
    }
    *out_acc = (double)ok / rows;
    return NN_OK;
}

const AccuracyDef AccuracyRegression = {"Regression", accuracy_regression};
const AccuracyDef AccuracyCategorical = {"Categorical", accuracy_categorical};
