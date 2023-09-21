#ifndef LOSS_FUNCS_H
#define LOSS_FUNCS_H

#include "tensor.h"

typedef struct LossDef
{
    const char *name;
    NNStatus (*value)(const Tensor *pred, const Tensor *truth, double *out_loss);
    NNStatus (*grad)(Tensor *d_pred, const Tensor *pred, const Tensor *truth);
} LossDef;

extern const LossDef LossMSE;
extern const LossDef LossBCE;
extern const LossDef LossCCE;

const LossDef *loss_by_name(const char *name);

#endif
