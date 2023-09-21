#ifndef TENSOR_H
#define TENSOR_H

#include "config.h"

typedef struct
{
    int n_dims;
    int *dims;
    int n_elements;
    TensorType *data;
} Tensor;

NNStatus tensor_create(Tensor *t, int n_dims, const int *dims);
void tensor_free(Tensor *t);
NNStatus tensor_fill(Tensor *t, TensorType value);
NNStatus tensor_copy(Tensor *dst, const Tensor *src);
NNStatus tensor_clone(Tensor *dst, const Tensor *src);
NNStatus tensor_get_index(const Tensor *t, const int *indices, int *out_index);
NNStatus tensor_get(const Tensor *t, const int *indices, TensorType *out_value);
NNStatus tensor_set(Tensor *t, const int *indices, TensorType value);
int tensor_argmax_row(const Tensor *t, int row);
NNStatus tensor_slice_row(Tensor *dst, const Tensor *src, int row);
void tensor_print(const Tensor *t);

#endif
