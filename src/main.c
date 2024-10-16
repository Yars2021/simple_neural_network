#include "neural_net.h"

#define NUM_OF_LAYERS   3

size_t layer_config[NUM_OF_LAYERS] = {49, 14, 3};

char *to_str(int x) {
    int len = 0;

    for (int i = x; i > 0; i /= 10, len++);

    char *buf = (char*)malloc(len + 1);
    buf[len] = '\0';

    for (size_t i = 0; i < len; i++, x /= 10) {
        buf[len - i - 1] = (char)(x % 10) + '0';
    }

    return buf;
}

int main() {
    DoubleVector_t *input_vector = create_double_vector(49), *expected_vector = create_double_vector(3);
    NeuralNet_t *net = create_neural_net(NUM_OF_LAYERS, layer_config);
    read_test(input_vector, expected_vector, "../input");

    set_input(net, input_vector);
    set_expected(net, expected_vector);

    train(net);
    printf("%zd\n", test(net));

    free_neural_net(net);
    free(input_vector);
    free(expected_vector);

    return 0;
}
