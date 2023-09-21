#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

static int count_total_elements(int n_dims, const int *dims)
{
    int total = 1;
    for (int i = 0; i < n_dims; i++)
    {
        if (dims[i] <= 0)
        {
            return 0;
        }
        total *= dims[i];
    }
    return total;
}

NNStatus tensor_create(Tensor *t, int n_dims, const int *dims)
{
    if (!t || !dims || n_dims <= 0)
    {
        return NN_ERR_INVALID_ARG;
    }

    memset(t, 0, sizeof(*t));
    int n_elements = count_total_elements(n_dims, dims);
    if (n_elements <= 0)
    {
        return NN_ERR_SHAPE;
    }

    t->dims = (int *)malloc((size_t)n_dims * sizeof(int));
    t->data = (TensorType *)malloc((size_t)n_elements * sizeof(TensorType));
    if (!t->dims || !t->data)
    {
        tensor_free(t);
        return NN_ERR_ALLOC;
    }

    t->n_dims = n_dims;
    t->n_elements = n_elements;
    memcpy(t->dims, dims, (size_t)n_dims * sizeof(int));
    return NN_OK;
}

void tensor_free(Tensor *t)
{
    if (!t)
    {
        return;
    }
    free(t->dims);
    free(t->data);
    t->dims = NULL;
    t->data = NULL;
    t->n_dims = 0;
    t->n_elements = 0;
}

NNStatus tensor_fill(Tensor *t, TensorType value)
{
    if (!t || !t->data)
    {
        return NN_ERR_INVALID_ARG;
    }
    for (int i = 0; i < t->n_elements; i++)
    {
        t->data[i] = value;
    }
    return NN_OK;
}

NNStatus tensor_copy(Tensor *dst, const Tensor *src)
{
    if (!dst || !src || !dst->data || !src->data || dst->n_dims != src->n_dims)
    {
        return NN_ERR_INVALID_ARG;
    }
    for (int i = 0; i < src->n_dims; i++)
    {
        if (dst->dims[i] != src->dims[i])
        {
            return NN_ERR_SHAPE;
        }
    }
    memcpy(dst->data, src->data, (size_t)src->n_elements * sizeof(TensorType));
    return NN_OK;
}

NNStatus tensor_clone(Tensor *dst, const Tensor *src)
{
    if (!dst || !src)
    {
        return NN_ERR_INVALID_ARG;
    }
    NNStatus st = tensor_create(dst, src->n_dims, src->dims);
    if (st != NN_OK)
    {
        return st;
    }
    return tensor_copy(dst, src);
}

NNStatus tensor_get_index(const Tensor *t, const int *indices, int *out_index)
{
    if (!t || !indices || !out_index || !t->dims)
    {
        return NN_ERR_INVALID_ARG;
    }

    int index = 0;
    int stride = 1;
    for (int i = t->n_dims - 1; i >= 0; i--)
    {
        if (indices[i] < 0 || indices[i] >= t->dims[i])
        {
            return NN_ERR_SHAPE;
        }
        index += indices[i] * stride;
        stride *= t->dims[i];
    }
    *out_index = index;
    return NN_OK;
}

NNStatus tensor_get(const Tensor *t, const int *indices, TensorType *out_value)
{
    int index = 0;
    NNStatus st = tensor_get_index(t, indices, &index);
    if (st != NN_OK)
    {
        return st;
    }
    *out_value = t->data[index];
    return NN_OK;
}

NNStatus tensor_set(Tensor *t, const int *indices, TensorType value)
{
    int index = 0;
    NNStatus st = tensor_get_index(t, indices, &index);
    if (st != NN_OK)
    {
        return st;
    }
    t->data[index] = value;
    return NN_OK;
}

int tensor_argmax_row(const Tensor *t, int row)
{
    if (!t || t->n_dims != 2 || row < 0 || row >= t->dims[0])
    {
        return -1;
    }
    int cols = t->dims[1];
    int offset = row * cols;
    int best = 0;
    TensorType best_v = t->data[offset];
    for (int c = 1; c < cols; c++)
    {
        TensorType v = t->data[offset + c];
        if (v > best_v)
        {
            best = c;
            best_v = v;
        }
    }
    return best;
}

NNStatus tensor_slice_row(Tensor *dst, const Tensor *src, int row)
{
    if (!dst || !src || src->n_dims != 2 || row < 0 || row >= src->dims[0])
    {
        return NN_ERR_INVALID_ARG;
    }
    int dims[1] = {src->dims[1]};
    NNStatus st = tensor_create(dst, 1, dims);
    if (st != NN_OK)
    {
        return st;
    }
    memcpy(dst->data, src->data + (row * src->dims[1]), (size_t)src->dims[1] * sizeof(TensorType));
    return NN_OK;
}

void tensor_print(const Tensor *t)
{
    if (!t || !t->data)
    {
        printf("<null tensor>\n");
        return;
    }

    if (t->n_dims == 2)
    {
        for (int r = 0; r < t->dims[0]; r++)
        {
            printf("[");
            for (int c = 0; c < t->dims[1]; c++)
            {
                printf("%.6f", t->data[r * t->dims[1] + c]);
                if (c + 1 < t->dims[1])
                {
                    printf(", ");
                }
            }
            printf("]\n");
        }
        return;
    }

    printf("[");
    for (int i = 0; i < t->n_elements; i++)
    {
        printf("%.6f", t->data[i]);
        if (i + 1 < t->n_elements)
        {
            printf(", ");
        }
    }
    printf("]\n");
}
