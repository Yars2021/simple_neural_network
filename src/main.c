#include "neural_net.h"
#include <string.h>

#define NUM_OF_LAYERS   3
#define NUM_OF_TRAIN    18
#define NUM_OF_TESTS    18

size_t layer_config[NUM_OF_LAYERS] = {49, 14, 3};

char *to_str(int x) {
    int len = (x == 0);

    for (int i = x; i > 0; i /= 10, len++);

    char *buf = (char*)malloc(len + 1);
    buf[len] = '\0';

    for (size_t i = 0; i < len; i++, x /= 10) {
        buf[len - i - 1] = (char)(x % 10) + '0';
    }

    return buf;
}

char *test_path(int test_id) {
    char *num = to_str(test_id);
    char *path = (char*)malloc(strlen("../data/") + strlen(num) + 1);

    memcpy(path, "../data/", 9);
    memcpy(path + 8, num, strlen(num) + 1);

    return path;
}

void train_on_set(NeuralNet_t *net) {
    DoubleVector_t *input_vector = create_double_vector(49), *expected_vector = create_double_vector(3);

    for (size_t i = 0; i < NUM_OF_TRAIN; i++) {
        char *path = test_path(i);

        read_test(input_vector, expected_vector, path);
        set_input(net, input_vector);
        set_expected(net, expected_vector);
        train(net);

        free(path);
    }

    free(input_vector);
    free(expected_vector);
}

void test_on_set(NeuralNet_t *net) {
    DoubleVector_t *input_vector = create_double_vector(49), *expected_vector = create_double_vector(3);

    for (size_t i = 0; i < NUM_OF_TRAIN; i++) {
        char *path = test_path(i);

        read_test(input_vector, expected_vector, path);
        set_input(net, input_vector);
        set_expected(net, expected_vector);
        printf("%s %zd\n - ", path, test(net));

        free(path);
    }

    free(input_vector);
    free(expected_vector);
}

int main() {

    NeuralNet_t *net = create_neural_net(NUM_OF_LAYERS, layer_config);

    train_on_set(net);
    test_on_set(net);

    free_neural_net(net);

    return 0;
}
