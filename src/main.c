#include <stdio.h>
#include <math.h>
#include "config.h"
#include "dataset.h"
#include "network.h"
#include "io.h"

static double compute_mae(const Tensor *pred, const Tensor *truth)
{
    double sum = 0.0;
    for (int i = 0; i < pred->n_elements; i++)
    {
        double d = pred->data[i] - truth->data[i];
        sum += fabs(d);
    }
    return sum / pred->n_elements;
}

static int run_categorical_demo(void)
{
    Dataset train = {0}, test = {0};
    if (dataset_load_csv_classification(&train, "data/categorical_train.csv", 2, 2) != NN_OK) return 1;
    if (dataset_load_csv_classification(&test, "data/categorical_test.csv", 2, 2) != NN_OK) { dataset_free(&train); return 1; }

    Network net = {0};
    if (network_create(&net, 2) != NN_OK) return 1;
    if (network_add_layer(&net, 2, 8, &ActivationTanh, -0.5, 0.5) != NN_OK) return 1;
    if (network_add_layer(&net, 8, 2, &ActivationSoftmax, -0.5, 0.5) != NN_OK) return 1;

    FitConfig cfg = {.epochs = 300, .batch_size = 8, .learning_rate = 0.1, .loss = &LossCCE, .accuracy = &AccuracyCategorical, .verbose = 1};
    if (network_fit(&net, &train.x, &train.y, &cfg) != NN_OK) return 1;

    double test_loss = 0.0, test_acc = 0.0;
    if (network_evaluate(&net, &test.x, &test.y, &LossCCE, &AccuracyCategorical, &test_loss, &test_acc) != NN_OK) return 1;
    printf("[Categorical] Test: loss=%.6f acc=%.6f\n", test_loss, test_acc);

    if (save_network("artifacts/categorical_model.nn", &net) != NN_OK) return 1;

    Network loaded = {0};
    if (load_network("artifacts/categorical_model.nn", &loaded) != NN_OK) return 1;

    Tensor p1 = {0}, p2 = {0};
    if (network_predict(&net, &test.x, &p1) != NN_OK) return 1;
    if (network_predict(&loaded, &test.x, &p2) != NN_OK) return 1;

    double max_diff = 0.0;
    for (int i = 0; i < p1.n_elements; i++)
    {
        double d = fabs(p1.data[i] - p2.data[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("[Categorical] Reload consistency max_diff=%.12f\n", max_diff);

    tensor_free(&p1);
    tensor_free(&p2);
    network_free(&loaded);
    network_free(&net);
    dataset_free(&train);
    dataset_free(&test);
    return max_diff < 1e-10 ? 0 : 1;
}

static int run_regression_demo(void)
{
    Dataset train = {0}, test = {0};
    if (dataset_load_csv_regression(&train, "data/regression_train.csv", 2, 1) != NN_OK) return 1;
    if (dataset_load_csv_regression(&test, "data/regression_test.csv", 2, 1) != NN_OK) { dataset_free(&train); return 1; }

    Network net = {0};
    if (network_create(&net, 1) != NN_OK) return 1;
    if (network_add_layer(&net, 2, 1, &ActivationIdentity, -0.2, 0.2) != NN_OK) return 1;

    FitConfig cfg = {.epochs = 400, .batch_size = 8, .learning_rate = 0.05, .loss = &LossMSE, .accuracy = &AccuracyRegression, .verbose = 1};
    if (network_fit(&net, &train.x, &train.y, &cfg) != NN_OK) return 1;

    double test_loss = 0.0, test_acc = 0.0;
    if (network_evaluate(&net, &test.x, &test.y, &LossMSE, &AccuracyRegression, &test_loss, &test_acc) != NN_OK) return 1;
    Tensor reg_pred = {0};
    if (network_predict(&net, &test.x, &reg_pred) != NN_OK) return 1;
    double reg_mae = compute_mae(&reg_pred, &test.y);
    printf("[Regression] Test: loss=%.6f mae=%.6f acc=%.6f\n", test_loss, reg_mae, test_acc);
    tensor_free(&reg_pred);

    if (save_network("artifacts/regression_model.nn", &net) != NN_OK) return 1;

    Network loaded = {0};
    if (load_network("artifacts/regression_model.nn", &loaded) != NN_OK) return 1;

    Tensor p1 = {0}, p2 = {0};
    if (network_predict(&net, &test.x, &p1) != NN_OK) return 1;
    if (network_predict(&loaded, &test.x, &p2) != NN_OK) return 1;

    double max_diff = 0.0;
    for (int i = 0; i < p1.n_elements; i++)
    {
        double d = fabs(p1.data[i] - p2.data[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("[Regression] Reload consistency max_diff=%.12f\n", max_diff);

    tensor_free(&p1);
    tensor_free(&p2);
    network_free(&loaded);
    network_free(&net);
    dataset_free(&train);
    dataset_free(&test);
    return max_diff < 1e-10 ? 0 : 1;
}

int main(void)
{
    nn_set_random_seed(42u);

    if (run_categorical_demo() != 0) return 1;
    if (run_regression_demo() != 0) return 1;
    return 0;
}
