#ifndef __NEURAL_NET
#define __NEURAL_NET

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MIN_ALPHA   0.1
#define MAX_ALPHA   0.3
#define MAX_ERROR   0.1

typedef struct {
    size_t size;
    double *values;
} DoubleVector_t;

DoubleVector_t *create_double_vector(size_t size);
void free_double_vector(DoubleVector_t *vector);
double get_value(DoubleVector_t *vector, size_t index);
void set_value(DoubleVector_t *vector, size_t index, double value);
double fold(DoubleVector_t *vector, double (*func)(double, double), double acc_start);
void print_vector(DoubleVector_t *vector);

typedef struct {
    double alpha;
    DoubleVector_t *expected;
    size_t num_of_layers;
    DoubleVector_t **layers;
    DoubleVector_t **deltas;
    DoubleVector_t ***weights;
} NeuralNet_t;

NeuralNet_t *create_neural_net(size_t num_of_layers, size_t *layer_sizes);
free_neural_net(NeuralNet_t *net);
void set_input(NeuralNet_t *net, DoubleVector_t *input);
void set_expected(NeuralNet_t *net, DoubleVector_t *expected);
void set_weights(NeuralNet_t *net, DoubleVector_t ***weights);
void clear(NeuralNet_t *net);
void calculate(NeuralNet_t *net);
double get_error(NeuralNet_t *net);
double get_alpha(NeuralNet_t *net);
void backtrack(NeuralNet_t *net);
void train(NeuralNet_t *net);
size_t test(NeuralNet_t *net);

#endif
