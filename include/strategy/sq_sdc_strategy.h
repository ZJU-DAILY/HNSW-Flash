#pragma once

#include "solve_strategy.h"
#include "../space/space_sq.h"
#include "../../third_party/hnswlib/hnswlib.h"

class SqSdcStrategy: public SolveStrategy {
public:
    SqSdcStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path):
        SolveStrategy(source_path, query_path, codebooks_path, index_path) {}

    void solve() {
        hnswlib::SqSdcSpace sq_space(data_dim_, data_set_);
        std::vector<std::vector<data_t>> encoded_data(data_num_);
        hnswlib::HierarchicalNSW<float> hnsw(&sq_space, data_num_, M_, ef_construction_);

        if (std::filesystem::exists(codebooks_path_)) {
            std::ifstream in(codebooks_path_, std::ios::binary);
            in.read(reinterpret_cast<char*>(&hnswlib::max_value_), sizeof(float));
            in.read(reinterpret_cast<char*>(&hnswlib::min_value_), sizeof(float));
            in.read(reinterpret_cast<char*>(&hnswlib::max_minus_min_value_), sizeof(float));
            in.read(reinterpret_cast<char*>(&hnswlib::alpha_), sizeof(float));
            hnsw.loadIndex(index_path_, &sq_space, data_num_);
        } else {
            // Encode data using SQ
            auto s_encode_data = std::chrono::system_clock::now();
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < data_num_; ++i) {
                encoded_data[i] = sq_space.encode_(data_set_[i]);
            }
            auto e_encode_data = std::chrono::system_clock::now();
            std::cout << "encode data cost: " << time_cost(s_encode_data, e_encode_data) << " (ms)\n";

            // Build HNSW index using encoded data
            auto s_build = std::chrono::system_clock::now();
            #pragma omp parallel for schedule(dynamic)
            for (uint32_t i = 0; i < data_num_; ++i) {
                hnsw.addPoint(encoded_data[i].data(), i);
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

            {
                std::filesystem::path fsPath(codebooks_path_);
                fsPath.remove_filename();
                std::filesystem::create_directories(fsPath);
                std::ofstream out(codebooks_path_, std::ios::binary);
                out.write(reinterpret_cast<char*>(&hnswlib::max_value_), sizeof(float));
                out.write(reinterpret_cast<char*>(&hnswlib::min_value_), sizeof(float));
                out.write(reinterpret_cast<char*>(&hnswlib::max_minus_min_value_), sizeof(float));
                out.write(reinterpret_cast<char*>(&hnswlib::alpha_), sizeof(float));
                hnsw.saveIndex(index_path_);
            }
        }

        // Encode query using PQ
        auto s_solve = std::chrono::system_clock::now();
        std::vector<std::vector<data_t>> encoded_query(query_num_);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < query_num_; ++i) {
            encoded_query[i] = sq_space.encode_(query_set_[i]);
        }
        hnsw.setEf(ef_search_);
        #pragma omp parallel for schedule(dynamic)
        for (uint32_t i = 0; i < query_num_; ++i) {
#if defined(RERANK)
            std::priority_queue<std::pair<float, hnswlib::labeltype>> tmp = hnsw.searchKnn(encoded_query[i].data(), K * 100);
            std::priority_queue<std::pair<float, hnswlib::labeltype>, std::vector<std::pair<float, hnswlib::labeltype>>, std::greater<>> result;

            while (!tmp.empty()) {
                float res = 0;
                size_t a = tmp.top().second;
                for (int j = 0; j < data_dim_; ++j) {
                    float t = org_data_set_[a][j] - org_query_set_[i][j];
                    res += t * t;
                }
                result.emplace(res, a);
                tmp.pop();
            }
#else
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw.searchKnn(encoded_query[i].data(), K);
#endif
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
    };
};
