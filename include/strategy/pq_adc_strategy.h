#pragma once

#include "solve_strategy.h"
#include "pq_sdc_strategy.h"
#include "../space/space_pq.h"
#include "../../third_party/hnswlib/hnswlib.h"

class PqAdcStrategy: public PqSdcStrategy {
public:
    PqAdcStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path):
        PqSdcStrategy(source_path, query_path, codebooks_path, index_path) {}

    void solve() {
        hnswlib::PqSdcSpace pq_sdc_space(this->subvector_num_);
        hnswlib::HierarchicalNSW<float> hnsw(&pq_sdc_space, this->data_num_, this->M_, this->ef_construction_);

        if (std::filesystem::exists(this->codebooks_path_)) {
            std::ifstream in(this->codebooks_path_, std::ios::binary);
            auto& codebooks = hnswlib::codebooks_;
            codebooks.resize(this->subvector_num_);
            for (int i = 0; i < this->subvector_num_; ++i) {
                codebooks[i].resize(CLUSTER_NUM);
                for (int j = 0; j < CLUSTER_NUM; ++j) {
                    codebooks[i][j].resize(SUBVECTOR_LENGTH);
                    for (int k = 0; k < SUBVECTOR_LENGTH; ++k) {
                        in.read(reinterpret_cast<char*>(&codebooks[i][j][k]), sizeof(float));
                    }
                }
            }
            auto& dist = hnswlib::dist_;
            dist.resize(this->subvector_num_);
            for (int i = 0; i < this->subvector_num_; ++i) {
                dist[i].resize(CLUSTER_NUM);
                for (int j = 0; j < CLUSTER_NUM; ++j) {
                    dist[i][j].resize(CLUSTER_NUM);
                    for (int k = 0; k < CLUSTER_NUM; ++k) {
                        in.read(reinterpret_cast<char*>(&dist[i][j][k]), sizeof(float));
                    }
                }
            }
            this->data_dim_ = this->subvector_num_;
            hnsw.loadIndex(this->index_path_, &pq_sdc_space, this->data_num_);
        } else {
            // Generate/Read codebooks
            auto s_gen = std::chrono::system_clock::now();
            this->generate_codebooks(this->sample_num_);
            auto e_gen = std::chrono::system_clock::now();
            std::cout << "generate codebooks cost: " << time_cost(s_gen, e_gen) << " (ms)\n";

            // Encode data using PQ
            auto s_encode_data = std::chrono::system_clock::now();
            std::vector<std::vector<data_t>> encoded_data(this->data_num_);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < this->data_num_; ++i) {
                encoded_data[i] = this->pqEncode(this->data_set_[i]);
            }
            auto e_encode_data = std::chrono::system_clock::now();
            std::cout << "encode data cost: " << time_cost(s_encode_data, e_encode_data) << " (ms)\n";

            // Build HNSW index using encoded data
            auto s_build = std::chrono::system_clock::now();
            this->data_num_ = encoded_data.size();
            this->data_dim_ = encoded_data[0].size();
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < this->data_num_; ++i) {
                hnsw.addPoint(encoded_data[i].data(), i);
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

            {
                std::filesystem::path fsPath(this->codebooks_path_);
                fsPath.remove_filename();
                std::filesystem::create_directories(fsPath);
                std::ofstream out(this->codebooks_path_, std::ios::binary);
                auto& codebooks = hnswlib::codebooks_;
                codebooks.resize(this->subvector_num_);
                for (int i = 0; i < this->subvector_num_; ++i) {
                    codebooks[i].resize(CLUSTER_NUM);
                    for (int j = 0; j < CLUSTER_NUM; ++j) {
                        codebooks[i][j].resize(SUBVECTOR_LENGTH);
                        for (int k = 0; k < SUBVECTOR_LENGTH; ++k) {
                            out.write(reinterpret_cast<char*>(&codebooks[i][j][k]), sizeof(float));
                        }
                    }
                }
                auto& dist = hnswlib::dist_;
                dist.resize(this->subvector_num_);
                for (int i = 0; i < this->subvector_num_; ++i) {
                    dist[i].resize(CLUSTER_NUM);
                    for (int j = 0; j < CLUSTER_NUM; ++j) {
                        dist[i][j].resize(CLUSTER_NUM);
                        for (int k = 0; k < CLUSTER_NUM; ++k) {
                            out.write(reinterpret_cast<char*>(&dist[i][j][k]), sizeof(float));
                        }
                    }
                }
                hnsw.saveIndex(this->index_path_);
            }
        }

        auto s_solve = std::chrono::system_clock::now();
        hnsw.fstdistfunc_ = hnswlib::PqAdcL2Sqr;
        hnsw.setEf(this->ef_search_);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < this->query_num_; ++i) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw.searchKnn(this->query_set_[i].data(), K);
            while (!result.empty() && this->knn_results_[i].size() < K) {
                this->knn_results_[i].emplace_back(result.top().second);
                result.pop();
            }
            while (this->knn_results_[i].size() < K) {
                this->knn_results_[i].emplace_back(-1);
            }
        }
        auto e_solve = std::chrono::system_clock::now();
        std::cout << "solve cost: " << time_cost(s_solve, e_solve) << " (ms)\n";
    };
};
