#pragma once

#include "solve_strategy.h"
#include "hnswlib/taumg.h"

class TauMgStrategy: public SolveStrategy {
public:
    TauMgStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path):
        SolveStrategy(source_path, query_path, codebooks_path, index_path) {
    }

    void solve() {
        hnswlib::L2Space l2space(data_dim_);
        hnswlib::TauMg<float> taumg(&l2space, data_num_, M_, ef_construction_);

        if (std::filesystem::exists(codebooks_path_)) {
            taumg.loadIndex(index_path_, &l2space, data_num_);
        } else {
            // Build index
            auto s_build = std::chrono::system_clock::now();

            std::vector<float> centroids(data_dim_, 0.0);
            // create NNG
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
            for (size_t i = 0; i < data_num_; ++i) {
                for (size_t j = 0; j < data_dim_; ++j) {
                    centroids[j] += data_set_[i][j];
                }
                taumg.addPoint(data_set_[i].data(), i);
            }
            for (size_t j = 0; j < data_dim_; ++j) {
                centroids[j] /= data_num_;
            }
            taumg.setCentroids(centroids.data());

#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
            for (int i = data_num_ - 1; i >= 0; --i) {
                taumg.taumgPrune(i);
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";
            {
                std::filesystem::path fsPath(codebooks_path_);
                fsPath.remove_filename();
                std::filesystem::create_directories(fsPath);
                std::ofstream out(codebooks_path_, std::ios::binary);
                taumg.saveIndex(index_path_);
            }
        }
        // search 
        auto s_solve = std::chrono::system_clock::now();
        taumg.setEf(EF_SEARCH);
#pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
        for (size_t i = 0; i < query_num_; ++i) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = taumg.searchKnn(query_set_[i].data(), K);
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
