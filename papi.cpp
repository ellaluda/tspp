#include <iostream>
#include <vector>
#include <limits>
#include "papi.h"
#include <unordered_map>
#include <algorithm>
#include <numeric>

class CSR_graph {
    int row_count; //number of vertices in graph
    unsigned int col_count; //number of edges in graph
    
    std::vector<unsigned int> row_ptr;
    std::vector<int> col_ids;
    std::vector<double> vals;
    
    std::unordered_map<int, double> calculated_weights;
public:

    void read(const char* filename) {
        FILE *graph_file = fopen(filename, "rb");
        fread(reinterpret_cast<char*>(&row_count), sizeof(int), 1, graph_file);
        fread(reinterpret_cast<char*>(&col_count), sizeof(unsigned int), 1, graph_file);

        std::cout << "Row_count = " << row_count << ", col_count = " << col_count << std::endl;
        
        row_ptr.resize(row_count + 1);
        col_ids.resize(col_count);
        vals.resize(col_count);
         
        fread(reinterpret_cast<char*>(row_ptr.data()), sizeof(unsigned int), row_count + 1, graph_file);
        fread(reinterpret_cast<char*>(col_ids.data()), sizeof(int), col_count, graph_file);
        fread(reinterpret_cast<char*>(vals.data()), sizeof(double), col_count, graph_file);
        fclose(graph_file);
    }

    void print_vertex(int idx) {
        for (int col = row_ptr[idx]; col < row_ptr[idx + 1]; col++) {
            std::cout << col_ids[col] << " " << vals[col] <<std::endl;
        }
        std::cout << std::endl;
    }

    void reset() {
        row_count = 0;
        col_count = 0;
        row_ptr.clear();
        col_ids.clear();
        vals.clear();
    }

    //Alg_1
    int find_max_weight_to_even_vertex(){
        std::vector<double> even_sums(row_count, 0.0);
        for (int i = 0; i < row_count; ++i)
        {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            {
                if (col_ids[j] % 2 == 0)
                {
                    even_sums[i] += vals[j];
                }
            }
        }
        auto max_it = std::max_element(even_sums.begin(), even_sums.end());
        return std::distance(even_sums.begin(), max_it);
    }

    //Alg_2
    double compute_neighbor_weight(int neighbor) {
        double neighbor_weight = 0.0;
        for (int k = row_ptr[neighbor]; k < row_ptr[neighbor + 1]; ++k) {
            int n = col_ids[k];
            neighbor_weight += vals[k] * (row_ptr[n + 1] - row_ptr[n]);
        }
        return neighbor_weight;
    }

    int find_max_rank_vertex() {
        double max_rank = -std::numeric_limits<double>::max();
        int max_vertex = INT_MIN;

        for (int i = 0; i < row_count; ++i) {
            double curr_rank = 0.0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                int neighbor = col_ids[j];
                double w_edge = vals[j];

                double neighbor_weight = compute_neighbor_weight(neighbor);
                curr_rank += w_edge * neighbor_weight;
            }
            if (curr_rank > max_rank) {
                max_rank = curr_rank;
                max_vertex = i;
            }
        }
        return max_vertex;
    }
};

#define N_TESTS 5

#define CHECK_PAPI_ERROR(call, msg) do { \
    int retval = (call); \
    if (retval != PAPI_OK) { \
        std::cerr << msg << " error: " << retval << std::endl; \
        return 1; \
    } \
} while (0)


int main () {
    const char* filenames[N_TESTS];
    filenames[0] = "synt";
    filenames[1] = "road_graph";
    filenames[2] = "stanford";
    filenames[3] = "youtube";
    filenames[4] = "syn_rmat";
    long long values[3];
    int Eventset = PAPI_NULL, code;

    // Initialize PAPI library
    CHECK_PAPI_ERROR(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT ? PAPI_OK : PAPI_EINVAL, "Library init");

    // Create event set
    CHECK_PAPI_ERROR(PAPI_create_eventset(&Eventset), "Event set creation");

    // Add L1 cache miss event
    CHECK_PAPI_ERROR(PAPI_event_name_to_code("PAPI_L1_TCM", &code), "L1_TCM code conversion");
    CHECK_PAPI_ERROR(PAPI_add_event(Eventset, code), "Add L1_TCM event");

    // Add L2 cache miss event
    CHECK_PAPI_ERROR(PAPI_add_event(Eventset, PAPI_L2_TCM), "Add L2_TCM event");

    // Add custom cache references event
    CHECK_PAPI_ERROR(PAPI_event_name_to_code("perf::PERF_COUNT_HW_CACHE_REFERENCES", &code), "Custom event code conversion");
    CHECK_PAPI_ERROR(PAPI_add_event(Eventset, code), "Add custom event");


    

    for (int n_test = 0; n_test < N_TESTS; n_test++) {
    CSR_graph a;
    a.read(filenames[n_test]);
    std::cout << filenames[n_test] << std::endl;

    // 开始计数
    CHECK_PAPI_ERROR(PAPI_start(Eventset), "PAPI_start");

    std::cout << "max vertex:" << a.find_max_weight_to_even_vertex() << std::endl;

    // 停止计数
    CHECK_PAPI_ERROR(PAPI_stop(Eventset, values), "PAPI_stop");

    std::cout << "alg1: L1_TCM: " << values[0] << " L2_TCM: " << values[1] << " custom: " << values[2] << std::endl;

    // 重置计数器
    CHECK_PAPI_ERROR(PAPI_reset(Eventset), "PAPI_reset");

    // 开始计数
    CHECK_PAPI_ERROR(PAPI_start(Eventset), "PAPI_start");

    std::cout << "max weight ver:" << a.find_max_rank_vertex() << std::endl;

    // 停止计数
    CHECK_PAPI_ERROR(PAPI_stop(Eventset, values), "PAPI_stop");

    std::cout << "alg2: L1_TCM: " << values[0] << " L2_TCM: " << values[1] << " PAPI_TOT_INS: " << values[2] << std::endl;
}

// 关闭PAPI库
PAPI_shutdown();

return 0;
}