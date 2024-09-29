#include <iostream>
#include <queue>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#include <chrono>

template <typename T>
class MyConcurrentQueue {
public:
    MyConcurrentQueue(size_t size) : maxSize(size), done(false) {
        pthread_mutex_init(&mutex, NULL);
        pthread_cond_init(&notFull, NULL);
        pthread_cond_init(&notEmpty, NULL);
    }

    ~MyConcurrentQueue() {
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&notFull);
        pthread_cond_destroy(&notEmpty);
    }

    void put(const T& item) {
        pthread_mutex_lock(&mutex);
        while (queue.size() == maxSize) {
            pthread_cond_wait(&notFull, &mutex);
        }
        queue.push(item);
        std::cout << "Put: " << item << std::endl;
        pthread_cond_signal(&notEmpty);
        pthread_mutex_unlock(&mutex);
    }

    T get() {
        pthread_mutex_lock(&mutex);
        while (queue.empty() && !done) {
            pthread_cond_wait(&notEmpty, &mutex);
        }
        if (queue.empty() && done) {
            pthread_mutex_unlock(&mutex);
            return -1; // Используйте специальное значение -1, чтобы указать, что очередь пуста и done
        }
        T item = queue.front();
        queue.pop();
        std::cout << "Get: " << item << std::endl;
        pthread_cond_signal(&notFull);
        pthread_mutex_unlock(&mutex);
        return item;
    }

    void set_done(int num_consumers) {
        pthread_mutex_lock(&mutex);
        done = true;
        for (int i = 0; i < num_consumers; ++i) {
            queue.push(-1); // Разместите сигнал завершения для каждого потребителя
        }
        pthread_cond_broadcast(&notEmpty);
        pthread_mutex_unlock(&mutex);
    }

    void reserve(int n) {
        pthread_mutex_lock(&mutex);
        maxSize = n;
        pthread_cond_broadcast(&notFull);
        pthread_mutex_unlock(&mutex);
    }

private:
    std::queue<T> queue;
    size_t maxSize;
    bool done;
    pthread_mutex_t mutex;
    pthread_cond_t notFull;
    pthread_cond_t notEmpty;
};

// Определение глобальной очереди
MyConcurrentQueue<int> my_queue(10);

void* producer_func(void* params) {
    for (int i = 0; i < 1000000; i++) {
        my_queue.put(i);
    }
    return NULL;
}

void* consumer_func(void* params) {
    while (true) {
        int a = my_queue.get();
        if (a == -1) {
            break;
        }
        std::cout << "Consumed: " << a << std::endl;
    }
    return NULL;
}

void test_case(int num_writers, int num_readers, size_t queue_size, std::vector<double>& times, int index) {
    MyConcurrentQueue<int> q(queue_size);
    my_queue = q;
    my_queue.reserve(queue_size); //Динамическая настройка емкости очереди
    std::vector<pthread_t> writers(num_writers), readers(num_readers);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_writers; ++i) {
        pthread_create(&writers[i], NULL, producer_func, NULL);
    }
    for (int i = 0; i < num_readers; ++i) {
        pthread_create(&readers[i], NULL, consumer_func, NULL);
    }
    for (int i = 0; i < num_writers; ++i) {
        pthread_join(writers[i], NULL);
    }
    my_queue.set_done(num_readers);
    for (int i = 0; i < num_readers; ++i) {
        pthread_join(readers[i], NULL);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    times[index] = duration.count();
    std::cout << "Test with " << num_writers << " producers and " << num_readers << " consumers completed in " << times[index] << " seconds." << std::endl;
}

int main() {
    std::vector<double> times(4); // Используется для записи времени выполнения каждого тестового примера
    std::cout << "Running test cases..." << std::endl;

    std::cout << "Test 1: {1 , N}" << std::endl;
    test_case(1, 4, 5, times, 0);

    std::cout << "Test 2: {N , 1}" << std::endl;
    test_case(4, 1, 5, times, 1);

    std::cout << "Test 3: {M , N}" << std::endl;
    test_case(2, 5, 5, times, 2);

    std::cout << "Test 4: {1 , 1}" << std::endl;
    test_case(1, 1, 5, times, 3);

    std::cout << "All test cases completed." << std::endl;

    for (size_t i = 0; i < times.size(); ++i) {
        std::cout << "Test" << i + 1 << " duration: " << times[i] << " seconds." << std::endl;
    }
    return 0;
}
