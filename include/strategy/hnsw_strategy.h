#pragma once

#include "solve_strategy.h"
#include "../../third_party/hnswlib/hnswlib.h"

class HnswStrategy: public SolveStrategy {
public:
    HnswStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path):
        SolveStrategy(source_path, query_path, codebooks_path, index_path) {
    }

    void solve() {
        // Build HNSW index
        hnswlib::L2Space l2space(data_dim_);
        hnswlib::HierarchicalNSW<float> hnsw(&l2space, data_num_, M_, ef_construction_);

        if (std::filesystem::exists(codebooks_path_)) {
            hnsw.loadIndex(index_path_, &l2space, data_num_);
        } else {
            auto s_build = std::chrono::system_clock::now();
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < data_num_; ++i) {
                hnsw.addPoint(data_set_[i].data(), i);
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

            {
                std::filesystem::path fsPath(codebooks_path_);
                fsPath.remove_filename();
                std::filesystem::create_directories(fsPath);
                std::ofstream out(codebooks_path_, std::ios::binary);
                std::cout << "save index: " + index_path_ << std::endl;
                hnsw.saveIndex(index_path_);
            }
        }

        // Solve query
        auto s_solve = std::chrono::system_clock::now();
        hnsw.setEf(ef_search_);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < query_num_; ++i) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw.searchKnn(query_set_[i].data(), K);
            while (!result.empty() && knn_results_[i].size() < K) {
                knn_results_[i].emplace_back(result.top().second);
                result.pop();
            }
            while (knn_results_[i].size() < K) {
                knn_results_[i].emplace_back(-1);
            }
        }
        auto e_solve = std::chrono::system_clock::now();
        std::cout << "solve cost: " << time_cost(s_solve, e_solve) << " (ms)\n";
    }
};
