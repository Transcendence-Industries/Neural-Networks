#ifndef CALC_H
#define CALC_H

#include "layers.h"

NNStatus layer_forward(Layer *l, const Tensor *inputs, Tensor *outputs);
NNStatus layer_backward(Layer *l, const Tensor *d_out, Tensor *d_inputs);
NNStatus layer_apply_sgd(Layer *l, TensorType lr, int batch_size);

#endif
