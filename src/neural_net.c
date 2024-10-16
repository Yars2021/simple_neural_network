#include "neural_net.h"

double func_sum(double x, double acc) {
    return acc + x;
}

double func_max(double x, double acc) {
    return (acc > x) ? acc : x;
}

double activation(double arg) {
    return 1 / (1 + exp(-arg));
}

DoubleVector_t *create_double_vector(size_t size) {
    if (size == 0) return NULL;

    DoubleVector_t *vector = (DoubleVector_t*)malloc(sizeof(DoubleVector_t));
    
    vector->size = size;
    vector->values = (double*)malloc(size * sizeof(double));

    return vector;
}

void free_double_vector(DoubleVector_t *vector) {
    if (!vector) return;

    vector->size = 0;
    free(vector->values);
    free(vector);
}

double get_value(DoubleVector_t *vector, size_t index) {
    if (!vector || index > vector->size - 1) return 0.0;

    return vector->values[index];
}

void set_value(DoubleVector_t *vector, size_t index, double value) {
    if (!vector || index > vector->size - 1) return;

    vector->values[index] = value;
}

double fold(DoubleVector_t *vector, double (*func)(double, double), double acc_start) {
    if (!vector || !func) return acc_start;

    double acc = acc_start;

    for (size_t i = 0; i < vector->size; i++) {
        acc = (*func)(vector->values[i], acc);
    }

    return acc;
}

void print_vector(DoubleVector_t *vector) {
    if (!vector) return;

    for (size_t i = 0; i < vector->size; i++) {
        printf("%lf\t", vector->values[i]);
    }

    printf("\n");
}

void read_test(DoubleVector_t *input, DoubleVector_t *expected, char *filepath) {
    if (!input || !expected || !filepath) return;

    FILE *file = fopen(filepath, "r");

    if (!file) return;

    size_t width, height, exp_size;

    fscanf(file, "%zd %zd %zd\n", &width, &height, &exp_size);

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            fscanf(file, "%lf", &(input->values[i * width + j]));
        }
    }

    for (size_t i = 0; i < exp_size; i++) {
        fscanf(file, "%lf", &(expected->values[i]));
    }

    fclose(file);
}

NeuralNet_t *create_neural_net(size_t num_of_layers, size_t *layer_sizes) {
    if (num_of_layers == 0 || !layer_sizes) return NULL;

    NeuralNet_t *net = (NeuralNet_t*)malloc(sizeof(NeuralNet_t));

    net->alpha = 0.0;
    net->expected = create_double_vector(layer_sizes[num_of_layers - 1]);
    net->num_of_layers = num_of_layers;
    net->layers = (DoubleVector_t**)malloc(num_of_layers * sizeof(DoubleVector_t*));
    net->deltas = (DoubleVector_t**)malloc(num_of_layers * sizeof(DoubleVector_t*));
    net->weights = (DoubleVector_t***)malloc(num_of_layers * sizeof(DoubleVector_t**));

    for (size_t i = 0; i < num_of_layers; i++) {
        net->layers[i] = create_double_vector(layer_sizes[i]);
        net->deltas[i] = create_double_vector(layer_sizes[i]);
    }

    srandom(time(NULL));

    for (size_t i = 0; i < num_of_layers - 1; i++) {
        net->weights[i] = (DoubleVector_t**)malloc(net->layers[i]->size * sizeof(DoubleVector_t*));
        for (size_t j = 0; j < net->layers[i + 1]->size; j++) {
            net->weights[i][j] = create_double_vector(net->layers[i + 1]->size);
            for (size_t k = 0; k < net->layers[i + 1]->size; k++) {
                set_value(net->weights[i][j], k, ((float)random() / (float)(RAND_MAX)) * 2 - 1);
            }
        }
    }

    return net;
}

void free_neural_net(NeuralNet_t *net) {
    if (!net) return;

    net->alpha = 0.0;
    free_double_vector(net->expected);

    for (size_t i = 0; i < net->num_of_layers - 1; i++) {
        for (size_t j = 0; j < net->layers[i]->size; j++) {
            free_double_vector(net->weights[i][j]);
        }
        free(net->weights[i]);
    }

    for (size_t i = 0; i < net->num_of_layers; i++) {
        free_double_vector(net->layers[i]);
        free_double_vector(net->deltas[i]);
    }

    net->num_of_layers = 0; 

    free(net->layers);
    free(net->deltas);
    free(net->weights);
    free(net);
}

void set_input(NeuralNet_t *net, DoubleVector_t *input) {
    if (!net || !input || net->layers[0]->size != input->size) return;

    for (size_t i = 0; i < input->size; i++) {
        set_value(net->layers[0], i, get_value(input, i));
    }
}

void set_expected(NeuralNet_t *net, DoubleVector_t *expected) {
    if (!net || !expected || net->expected->size != expected->size) return;

    for (size_t i = 0; i < expected->size; i++) {
        set_value(net->expected, i, get_value(expected, i));
    }
}

void set_weights(NeuralNet_t *net, DoubleVector_t ***weights) {
    if (!net || !weights) return;

    for (size_t i = 0; i < net->num_of_layers - 1; i++) {
        for (size_t j = 0; j < net->layers[i + 1]->size; j++) {
            for (size_t k = 0; k < net->layers[i + 1]->size; k++) {
                set_value(net->weights[i][j], k, get_value(weights[i][j], k));
            }
        }
    }
}


void clear(NeuralNet_t *net) {
    if (!net) return;

    for (size_t i = 0; i < net->num_of_layers; i++) {
        for (size_t j = 0; j < net->layers[i]->size; j++) {
            set_value(net->layers[i], j, 0.0);
        }
    }
}

void calculate(NeuralNet_t *net) {
    if (!net) return;

    for (size_t i = 1; i < net->num_of_layers; i++) {
        for (size_t j = 0; j < net->layers[i]->size; j++) {
            double sum = 0;
            for (size_t k = 0; k < net->layers[i - 1]->size; k++) {
                sum += get_value(net->layers[i - 1], k) * get_value(net->weights[i - 1][k], j);
            }
            set_value(net->layers[i], j, activation(sum));
        }
    }
}

double get_error(NeuralNet_t *net) {
    if (!net) return 0.0;

    double error = 0;

    for (size_t i = 0; i < net->expected->size; i++) {
        error += fabs(get_value(net->expected, i) - get_value(net->layers[net->num_of_layers - 1], i));
    }

    return error / 2;
}

double get_alpha(NeuralNet_t *net) {
    if (!net) return 0.0;

    return (2 * get_error(net) / (double)(net->layers[net->num_of_layers - 1]->size)) * (MAX_ALPHA - MIN_ALPHA) + MIN_ALPHA;
}

void backtrack(NeuralNet_t *net) {
    if (!net) return;

    for (size_t i = 0; i < net->layers[net->num_of_layers - 1]->size; i++) {
        double t = get_value(net->expected, i), y = get_value(net->layers[net->num_of_layers - 1], i);
        set_value(net->deltas[net->num_of_layers - 1], i, y * (1 - y) * (t - y));
    }

    for (int i = (int)net->num_of_layers - 2; i >= 0; i--) {
        for (size_t j = 0; j < net->layers[i]->size; j++) {
            double y = get_value(net->layers[i], j), sum = 0;
            for (size_t k = 0; k < net->layers[i + 1]->size; k++) {
                sum += get_value(net->deltas[i + 1], k) * get_value(net->weights[i][j], k);
            }
            set_value(net->deltas[i], j, y * (1 - y) * sum);
        }
    }

    for (size_t i = 0; i < net->num_of_layers - 1; i++) {
        for (size_t j = 0; j < net->layers[i]->size; j++) {
            for (size_t k = 0; k < net->layers[i + 1]->size; k++) {
                double y = (i == 0) ? get_value(net->layers[0], j) : get_value(net->layers[i - 1], j);
                set_value(net->weights[i][j], k, get_value(net->weights[i][j], k) + net->alpha * get_value(net->deltas[i], j) * y);
            }
        }
    }
}

void train(NeuralNet_t *net) {
    if (!net) return;

    clear(net);
    calculate(net);
    net->alpha = get_alpha(net);

    if (get_error(net) > MAX_ERROR) {
        backtrack(net);
    }
}

size_t test(NeuralNet_t *net) {
    if (!net) return 0;

    clear(net);
    calculate(net);
    
    size_t ans = 0;

    for (size_t i = 0; i < net->layers[net->num_of_layers - 1]->size; i++) {
        if (get_value(net->layers[net->num_of_layers - 1], ans) < get_value(net->layers[net->num_of_layers - 1], i)) {
            ans = i;
        }
    }

    return ans;
}
