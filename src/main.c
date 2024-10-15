#include "neural_net.h"

double sum(double x, double acc) {
    return acc + x;
}

int main() {
    DoubleVector_t *vector = create_double_vector(10);

    set_value(vector, 0, 1);
    set_value(vector, 1, 2);
    set_value(vector, 2, 3);
    set_value(vector, 3, 4);
    set_value(vector, 4, 5);
    set_value(vector, 5, 6);
    set_value(vector, 6, 7);
    set_value(vector, 7, 8);
    set_value(vector, 8, 9);
    set_value(vector, 9, 10);

    print_vector(vector);

    printf("%f\n", fold(vector, &sum, 0));

    free_double_vector(vector);

    return 0;
}
