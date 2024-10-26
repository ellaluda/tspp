#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int random_walk(int a, int b, int x, double p) {
    while (x > a && x < b) {
        if (((double) rand() / RAND_MAX) < p) {
            x++;
        } else {
            x--;
        }
    }
    return x;
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        printf("Usage: %s a b x p N P\n", argv[0]);
        return 1;
    }

    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int x = atoi(argv[3]);
    double p = atof(argv[4]);
    int N = atoi(argv[5]);
    int P = atoi(argv[6]);

    int hits_b = 0;
    double total_lifetime = 0.0;

    srand(time(NULL));

    double start_time = omp_get_wtime();

    #pragma omp parallel num_threads(P)
        {
            unsigned int seed = time(NULL) ^ omp_get_thread_num();
    #pragma omp for reduction(+:hits_b, total_lifetime)
            for (int i = 0; i < N; i++) {
                int pos = x;
                int lifetime = 0;

                while (pos > a && pos < b) {
                    lifetime++;
                    if (((double) rand_r(&seed) / RAND_MAX) < p) {
                        pos++;
                    } else {
                        pos--;
                    }
                }

                if (pos == b) {
                    hits_b++;
                }
                total_lifetime += lifetime;
            }
        }



    double end_time = omp_get_wtime();
    double probability_b = (double) hits_b / N;
    double average_lifetime = total_lifetime / N;
    double execution_time = end_time - start_time;

    printf("Probability of reaching b: %f\n", probability_b);
    printf("Average lifetime: %f\n", average_lifetime);
    printf("Execution time: %f seconds\n", execution_time);

    return 0;
}
