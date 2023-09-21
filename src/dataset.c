#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dataset.h"

void dataset_free(Dataset *ds)
{
    if (!ds) return;
    tensor_free(&ds->x);
    tensor_free(&ds->y);
}

NNStatus dataset_load_csv_classification(Dataset *ds, const char *path, int n_features, int n_classes)
{
    if (!ds || !path || n_features <= 0 || n_classes <= 1) return NN_ERR_INVALID_ARG;
    memset(ds, 0, sizeof(*ds));

    FILE *f = fopen(path, "r");
    if (!f) return NN_ERR_IO;

    char line[1024];
    int rows = 0;
    while (fgets(line, sizeof(line), f))
    {
        if (line[0] != '\n' && line[0] != '\0') rows++;
    }
    if (rows <= 0) { fclose(f); return NN_ERR_FORMAT; }

    NNStatus st = tensor_create(&ds->x, 2, (int[]){rows, n_features});
    if (st != NN_OK) { fclose(f); return st; }
    st = tensor_create(&ds->y, 2, (int[]){rows, n_classes});
    if (st != NN_OK) { fclose(f); tensor_free(&ds->x); return st; }
    tensor_fill(&ds->y, 0.0);

    // Second pass parses values into tensors after row count is known.
    rewind(f);
    int row = 0;
    while (fgets(line, sizeof(line), f))
    {
        char *save = NULL;
        char *tok = strtok_r(line, ",", &save);
        int col = 0;
        while (tok && col < n_features)
        {
            ds->x.data[row * n_features + col] = (TensorType)strtod(tok, NULL);
            col++;
            tok = strtok_r(NULL, ",", &save);
        }
        if (col != n_features || !tok)
        {
            fclose(f);
            dataset_free(ds);
            return NN_ERR_FORMAT;
        }

        // Final column is an integer class id, expanded into a one-hot target row.
        int cls = atoi(tok);
        if (cls < 0 || cls >= n_classes)
        {
            fclose(f);
            dataset_free(ds);
            return NN_ERR_FORMAT;
        }
        ds->y.data[row * n_classes + cls] = 1.0;
        row++;
    }

    fclose(f);
    return NN_OK;
}

NNStatus dataset_load_csv_regression(Dataset *ds, const char *path, int n_features, int n_targets)
{
    if (!ds || !path || n_features <= 0 || n_targets <= 0) return NN_ERR_INVALID_ARG;
    memset(ds, 0, sizeof(*ds));

    FILE *f = fopen(path, "r");
    if (!f) return NN_ERR_IO;

    char line[1024];
    int rows = 0;
    while (fgets(line, sizeof(line), f))
    {
        if (line[0] != '\n' && line[0] != '\0') rows++;
    }
    if (rows <= 0) { fclose(f); return NN_ERR_FORMAT; }

    NNStatus st = tensor_create(&ds->x, 2, (int[]){rows, n_features});
    if (st != NN_OK) { fclose(f); return st; }
    st = tensor_create(&ds->y, 2, (int[]){rows, n_targets});
    if (st != NN_OK) { fclose(f); tensor_free(&ds->x); return st; }

    // Second pass parses values into tensors after row count is known.
    rewind(f);
    int row = 0;
    while (fgets(line, sizeof(line), f))
    {
        char *save = NULL;
        char *tok = strtok_r(line, ",", &save);
        int col = 0;
        while (tok && col < n_features)
        {
            ds->x.data[row * n_features + col] = (TensorType)strtod(tok, NULL);
            col++;
            tok = strtok_r(NULL, ",", &save);
        }
        if (col != n_features)
        {
            fclose(f);
            dataset_free(ds);
            return NN_ERR_FORMAT;
        }

        int t = 0;
        while (tok && t < n_targets)
        {
            ds->y.data[row * n_targets + t] = (TensorType)strtod(tok, NULL);
            t++;
            tok = strtok_r(NULL, ",", &save);
        }
        if (t != n_targets)
        {
            fclose(f);
            dataset_free(ds);
            return NN_ERR_FORMAT;
        }

        row++;
    }

    fclose(f);
    return NN_OK;
}
