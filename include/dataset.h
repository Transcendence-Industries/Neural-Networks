#ifndef DATASET_H
#define DATASET_H

#include "tensor.h"

typedef struct
{
    Tensor x;
    Tensor y;
} Dataset;

NNStatus dataset_load_csv_classification(Dataset *ds, const char *path, int n_features, int n_classes);
NNStatus dataset_load_csv_regression(Dataset *ds, const char *path, int n_features, int n_targets);
void dataset_free(Dataset *ds);

#endif
