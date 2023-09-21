#ifndef ACCURACY_FUNCS_H
#define ACCURACY_FUNCS_H

#include "tensor.h"

typedef struct AccuracyDef
{
    const char *name;
    NNStatus (*value)(const Tensor *pred, const Tensor *truth, double *out_acc);
} AccuracyDef;

extern const AccuracyDef AccuracyRegression;
extern const AccuracyDef AccuracyCategorical;

#endif
