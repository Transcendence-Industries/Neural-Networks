#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "io.h"

#define NET_MAGIC 0x4E4E4331u
#define NET_VERSION 1u

static NNStatus write_u32(FILE *f, uint32_t v) { return fwrite(&v, sizeof(v), 1, f) == 1 ? NN_OK : NN_ERR_IO; }
static NNStatus write_i32(FILE *f, int32_t v) { return fwrite(&v, sizeof(v), 1, f) == 1 ? NN_OK : NN_ERR_IO; }
static NNStatus read_u32(FILE *f, uint32_t *v) { return fread(v, sizeof(*v), 1, f) == 1 ? NN_OK : NN_ERR_IO; }
static NNStatus read_i32(FILE *f, int32_t *v) { return fread(v, sizeof(*v), 1, f) == 1 ? NN_OK : NN_ERR_IO; }

NNStatus save_network(const char *path, const Network *net)
{
    if (!path || !net) return NN_ERR_INVALID_ARG;
    FILE *f = fopen(path, "wb");
    if (!f) return NN_ERR_IO;

    NNStatus st = write_u32(f, NET_MAGIC);
    if (st == NN_OK) st = write_u32(f, NET_VERSION);
    if (st == NN_OK) st = write_i32(f, net->n_layers);

    for (int i = 0; st == NN_OK && i < net->n_layers; i++)
    {
        const Layer *l = &net->layers[i];
        uint32_t name_len = (uint32_t)(strlen(l->activation->name) + 1);
        st = write_i32(f, l->n_inputs);
        if (st == NN_OK) st = write_i32(f, l->n_neurons);
        if (st == NN_OK) st = write_u32(f, name_len);
        if (st == NN_OK && fwrite(l->activation->name, 1, name_len, f) != name_len) st = NN_ERR_IO;
        if (st == NN_OK && fwrite(l->weights.data, sizeof(TensorType), (size_t)l->weights.n_elements, f) != (size_t)l->weights.n_elements) st = NN_ERR_IO;
        if (st == NN_OK && fwrite(l->biases.data, sizeof(TensorType), (size_t)l->biases.n_elements, f) != (size_t)l->biases.n_elements) st = NN_ERR_IO;
    }

    fclose(f);
    return st;
}

NNStatus load_network(const char *path, Network *net)
{
    if (!path || !net) return NN_ERR_INVALID_ARG;
    FILE *f = fopen(path, "rb");
    if (!f) return NN_ERR_IO;

    network_free(net);

    uint32_t magic = 0, ver = 0;
    int32_t n_layers = 0;
    NNStatus st = read_u32(f, &magic);
    if (st == NN_OK) st = read_u32(f, &ver);
    if (st == NN_OK) st = read_i32(f, &n_layers);
    if (st != NN_OK || magic != NET_MAGIC || ver != NET_VERSION || n_layers <= 0)
    {
        fclose(f);
        return NN_ERR_FORMAT;
    }

    st = network_create(net, n_layers);
    if (st != NN_OK) { fclose(f); return st; }

    for (int i = 0; st == NN_OK && i < n_layers; i++)
    {
        int32_t n_inputs = 0, n_neurons = 0;
        uint32_t name_len = 0;
        st = read_i32(f, &n_inputs);
        if (st == NN_OK) st = read_i32(f, &n_neurons);
        if (st == NN_OK) st = read_u32(f, &name_len);
        if (st != NN_OK || name_len < 2 || name_len > 64) { st = NN_ERR_FORMAT; break; }

        char name[65];
        if (fread(name, 1, name_len, f) != name_len) { st = NN_ERR_IO; break; }
        name[name_len - 1] = '\0';

        const ActivationDef *act = activation_by_name(name);
        if (!act) { st = NN_ERR_FORMAT; break; }

        st = network_add_layer(net, n_inputs, n_neurons, act, 0.0, 0.0);
        if (st != NN_OK) break;
        Layer *l = &net->layers[i];

        if (fread(l->weights.data, sizeof(TensorType), (size_t)l->weights.n_elements, f) != (size_t)l->weights.n_elements) { st = NN_ERR_IO; break; }
        if (fread(l->biases.data, sizeof(TensorType), (size_t)l->biases.n_elements, f) != (size_t)l->biases.n_elements) { st = NN_ERR_IO; break; }
    }

    fclose(f);
    if (st != NN_OK) network_free(net);
    return st;
}
