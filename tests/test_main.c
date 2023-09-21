#include <stdio.h>
#include <math.h>
#include <string.h>
#include "config.h"
#include "tensor.h"
#include "activation_funcs.h"
#include "loss_funcs.h"
#include "dataset.h"
#include "network.h"
#include "io.h"

#define ASSERT_TRUE(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } } while (0)

static int test_tensor_basics(void)
{
    Tensor t = {0};
    ASSERT_TRUE(tensor_create(&t, 2, (int[]){2, 3}) == NN_OK, "tensor_create");
    ASSERT_TRUE(tensor_set(&t, (int[]){1, 2}, 7.0) == NN_OK, "tensor_set");
    TensorType v = 0.0;
    ASSERT_TRUE(tensor_get(&t, (int[]){1, 2}, &v) == NN_OK, "tensor_get");
    ASSERT_TRUE(fabs(v - 7.0) < 1e-12, "tensor value");
    tensor_free(&t);
    return 0;
}

static int test_activation_and_loss(void)
{
    Tensor in = {0}, out = {0}, truth = {0}, grad = {0};
    ASSERT_TRUE(tensor_create(&in, 2, (int[]){1, 2}) == NN_OK, "in create");
    ASSERT_TRUE(tensor_create(&out, 2, (int[]){1, 2}) == NN_OK, "out create");
    ASSERT_TRUE(tensor_create(&truth, 2, (int[]){1, 2}) == NN_OK, "truth create");
    ASSERT_TRUE(tensor_create(&grad, 2, (int[]){1, 2}) == NN_OK, "grad create");

    in.data[0] = 2.0;
    in.data[1] = 0.0;
    truth.data[0] = 1.0;
    truth.data[1] = 0.0;

    ASSERT_TRUE(ActivationSoftmax.forward(&out, &in) == NN_OK, "softmax");
    ASSERT_TRUE(out.data[0] > out.data[1], "softmax ordering");

    double loss = 0.0;
    ASSERT_TRUE(LossCCE.value(&out, &truth, &loss) == NN_OK, "cce value");
    ASSERT_TRUE(loss > 0.0, "cce positive");
    ASSERT_TRUE(LossCCE.grad(&grad, &out, &truth) == NN_OK, "cce grad");

    tensor_free(&in); tensor_free(&out); tensor_free(&truth); tensor_free(&grad);
    return 0;
}

static int test_dataset_and_pipeline(void)
{
    nn_set_random_seed(42u);

    Dataset train = {0}, test = {0};
    ASSERT_TRUE(dataset_load_csv_classification(&train, "data/categorical_train.csv", 2, 2) == NN_OK, "load train");
    ASSERT_TRUE(dataset_load_csv_classification(&test, "data/categorical_test.csv", 2, 2) == NN_OK, "load test");

    Network net = {0};
    ASSERT_TRUE(network_create(&net, 2) == NN_OK, "network create");
    ASSERT_TRUE(network_add_layer(&net, 2, 8, &ActivationTanh, -0.5, 0.5) == NN_OK, "l1");
    ASSERT_TRUE(network_add_layer(&net, 8, 2, &ActivationSoftmax, -0.5, 0.5) == NN_OK, "l2");

    double before_loss = 0.0, before_acc = 0.0;
    ASSERT_TRUE(network_evaluate(&net, &train.x, &train.y, &LossCCE, &AccuracyCategorical, &before_loss, &before_acc) == NN_OK, "evaluate before");

    FitConfig cfg = {.epochs = 200, .batch_size = 8, .learning_rate = 0.1, .loss = &LossCCE, .accuracy = &AccuracyCategorical, .verbose = 0};
    ASSERT_TRUE(network_fit(&net, &train.x, &train.y, &cfg) == NN_OK, "fit");

    double after_loss = 0.0, after_acc = 0.0;
    ASSERT_TRUE(network_evaluate(&net, &train.x, &train.y, &LossCCE, &AccuracyCategorical, &after_loss, &after_acc) == NN_OK, "evaluate after");
    ASSERT_TRUE(after_loss < before_loss, "loss decreased");

    double test_loss = 0.0, test_acc = 0.0;
    ASSERT_TRUE(network_evaluate(&net, &test.x, &test.y, &LossCCE, &AccuracyCategorical, &test_loss, &test_acc) == NN_OK, "eval test");
    ASSERT_TRUE(test_acc >= 0.65, "test baseline acc");
    ASSERT_TRUE(test_acc < 1.0, "test should not be perfect");

    ASSERT_TRUE(save_network("artifacts/test_categorical_model.nn", &net) == NN_OK, "save network");
    Network loaded = {0};
    ASSERT_TRUE(load_network("artifacts/test_categorical_model.nn", &loaded) == NN_OK, "load network");

    Tensor p1 = {0}, p2 = {0};
    ASSERT_TRUE(network_predict(&net, &test.x, &p1) == NN_OK, "predict orig");
    ASSERT_TRUE(network_predict(&loaded, &test.x, &p2) == NN_OK, "predict loaded");
    ASSERT_TRUE(p1.n_elements == p2.n_elements, "same pred shape");
    for (int i = 0; i < p1.n_elements; i++) ASSERT_TRUE(fabs(p1.data[i] - p2.data[i]) < 1e-10, "pred consistency");

    tensor_free(&p1); tensor_free(&p2);
    network_free(&loaded);
    network_free(&net);
    dataset_free(&train);
    dataset_free(&test);
    return 0;
}

static int test_regression_dataset_and_pipeline(void)
{
    nn_set_random_seed(42u);

    Dataset train = {0}, test = {0};
    ASSERT_TRUE(dataset_load_csv_regression(&train, "data/regression_train.csv", 2, 1) == NN_OK, "load reg train");
    ASSERT_TRUE(dataset_load_csv_regression(&test, "data/regression_test.csv", 2, 1) == NN_OK, "load reg test");

    Network net = {0};
    ASSERT_TRUE(network_create(&net, 1) == NN_OK, "reg network create");
    ASSERT_TRUE(network_add_layer(&net, 2, 1, &ActivationIdentity, -0.2, 0.2) == NN_OK, "reg l1");

    double before_loss = 0.0, before_acc = 0.0;
    ASSERT_TRUE(network_evaluate(&net, &train.x, &train.y, &LossMSE, &AccuracyRegression, &before_loss, &before_acc) == NN_OK, "reg evaluate before");

    FitConfig cfg = {.epochs = 400, .batch_size = 8, .learning_rate = 0.05, .loss = &LossMSE, .accuracy = &AccuracyRegression, .verbose = 0};
    ASSERT_TRUE(network_fit(&net, &train.x, &train.y, &cfg) == NN_OK, "reg fit");

    double after_loss = 0.0, after_acc = 0.0;
    ASSERT_TRUE(network_evaluate(&net, &train.x, &train.y, &LossMSE, &AccuracyRegression, &after_loss, &after_acc) == NN_OK, "reg evaluate after");
    ASSERT_TRUE(after_loss < before_loss, "reg loss decreased");

    double test_loss = 0.0, test_acc = 0.0;
    ASSERT_TRUE(network_evaluate(&net, &test.x, &test.y, &LossMSE, &AccuracyRegression, &test_loss, &test_acc) == NN_OK, "reg eval test");
    ASSERT_TRUE(test_acc >= 0.75 && test_acc <= 0.95, "reg test acc range");

    ASSERT_TRUE(save_network("artifacts/test_regression_model.nn", &net) == NN_OK, "save reg network");
    Network loaded = {0};
    ASSERT_TRUE(load_network("artifacts/test_regression_model.nn", &loaded) == NN_OK, "load reg network");

    Tensor p1 = {0}, p2 = {0};
    ASSERT_TRUE(network_predict(&net, &test.x, &p1) == NN_OK, "reg predict orig");
    ASSERT_TRUE(network_predict(&loaded, &test.x, &p2) == NN_OK, "reg predict loaded");
    ASSERT_TRUE(p1.n_elements == p2.n_elements, "reg same pred shape");
    for (int i = 0; i < p1.n_elements; i++) ASSERT_TRUE(fabs(p1.data[i] - p2.data[i]) < 1e-10, "reg pred consistency");

    tensor_free(&p1); tensor_free(&p2);
    network_free(&loaded);
    network_free(&net);
    dataset_free(&train);
    dataset_free(&test);
    return 0;
}

int main(void)
{
    if (test_tensor_basics()) return 1;
    if (test_activation_and_loss()) return 1;
    if (test_dataset_and_pipeline()) return 1;
    if (test_regression_dataset_and_pipeline()) return 1;
    printf("All tests passed.\n");
    return 0;
}
