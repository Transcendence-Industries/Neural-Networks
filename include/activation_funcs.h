#ifndef ACTIVATION_FUNCS_H
#define ACTIVATION_FUNCS_H

#include "tensor.h"

typedef struct ActivationDef
{
    const char *name;
    NNStatus (*forward)(Tensor *out, const Tensor *in);
    NNStatus (*backward)(Tensor *d_out, const Tensor *activated_out);
} ActivationDef;

extern const ActivationDef ActivationIdentity;
extern const ActivationDef ActivationRelu;
extern const ActivationDef ActivationSigmoid;
extern const ActivationDef ActivationTanh;
extern const ActivationDef ActivationSoftmax;

const ActivationDef *activation_by_name(const char *name);

#endif
