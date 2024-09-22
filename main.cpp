#include <iostream>
#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <sys/time.h>

double pi = 0.0;
int num_intervals;
int num_threads;
pthread_mutex_t mutex;

void* calculate_pi(void* arg) {
    int thread_id = *(int*)arg;
    double width = 1.0 / num_intervals;
    double local_sum = 0.0;

    for (int i = thread_id; i < num_intervals; i += num_threads) {
        double x = (i + 0.5) * width;
        local_sum += 4.0 / (1.0 + x * x);
    }

    pthread_mutex_lock(&mutex);
    pi += local_sum * width;
    pthread_mutex_unlock(&mutex);

    delete (int*)arg;
    return nullptr;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num-partition-intervals> <num-threads>" << std::endl;
        return 1;
    }

    num_intervals = std::atoi(argv[1]);
    num_threads = std::atoi(argv[2]);

    pthread_t threads[num_threads];
    pthread_mutex_init(&mutex, nullptr);

    struct timeval start, end;
    gettimeofday(&start, nullptr);

    for (int i = 0; i < num_threads; i++) {
        int *thread_id = new int(i);
        pthread_create(&threads[i], nullptr, calculate_pi, thread_id);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    gettimeofday(&end, nullptr);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << pi << std::endl;
    std::cout << "Elapsed time: " << elapsed_time << " s" << std::endl;

    pthread_mutex_destroy(&mutex);
    return 0;
}
