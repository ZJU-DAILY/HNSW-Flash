#pragma once

#include "solve_strategy.h"
#include "../space/space_pq.h"
#include "../../third_party/hnswlib/hnswlib.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class PqSdcStrategy: public SolveStrategy {
public:
    PqSdcStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string dist_path):
        SolveStrategy(source_path, query_path, codebooks_path, dist_path) {
        subvector_num_ = SUBVECTOR_NUM;
        assert(data_dim_ % subvector_num_ == 0);
        subvector_length_ = SUBVECTOR_LENGTH = data_dim_ / subvector_num_;
        cluster_num_ = CLUSTER_NUM;
        max_iterations_ = MAX_ITERATIONS;
        sample_num_ = std::min(SAMPLE_NUM, (size_t)data_num_);
    }

    void solve() {
        hnswlib::PqSdcSpace pq_sdc_space(PRINCIPAL_DIM);
        hnswlib::HierarchicalNSW<float> hnsw(&pq_sdc_space, data_num_, M_, ef_construction_);

        if (std::filesystem::exists(codebooks_path_)) {
            std::ifstream in(codebooks_path_, std::ios::binary);
            auto& codebooks = hnswlib::codebooks_;
            codebooks.resize(subvector_num_);
            for (int i = 0; i < subvector_num_; ++i) {
                codebooks[i].resize(CLUSTER_NUM);
                for (int j = 0; j < CLUSTER_NUM; ++j) {
                    codebooks[i][j].resize(SUBVECTOR_LENGTH);
                    for (int k = 0; k < SUBVECTOR_LENGTH; ++k) {
                        in.read(reinterpret_cast<char*>(&codebooks[i][j][k]), sizeof(float));
                    }
                }
            }
            auto& dist = hnswlib::dist_;
            dist.resize(subvector_num_);
            for (int i = 0; i < subvector_num_; ++i) {
                dist[i].resize(CLUSTER_NUM);
                for (int j = 0; j < CLUSTER_NUM; ++j) {
                    dist[i][j].resize(CLUSTER_NUM);
                    for (int k = 0; k < CLUSTER_NUM; ++k) {
                        in.read(reinterpret_cast<char*>(&dist[i][j][k]), sizeof(float));
                    }
                }
            }
            data_dim_ = subvector_num_;
            hnsw.loadIndex(index_path_, &pq_sdc_space, data_num_);
        } else {
            // Generate/Read codebooks
            auto s_gen = std::chrono::system_clock::now();
            generate_codebooks(sample_num_);
            auto e_gen = std::chrono::system_clock::now();
            std::cout << "generate codebooks cost: " << time_cost(s_gen, e_gen) << " (ms)\n";

            // Encode data using PQ
            auto s_encode_data = std::chrono::system_clock::now();
            std::vector<std::vector<data_t>> encoded_data(data_num_);
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < data_num_; ++i) {
                encoded_data[i] = pqEncode(data_set_[i]);
            }
            auto e_encode_data = std::chrono::system_clock::now();
            std::cout << "encode data cost: " << time_cost(s_encode_data, e_encode_data) << " (ms)\n";

            // Build HNSW index using encoded data
            auto s_build = std::chrono::system_clock::now();
            data_dim_ = encoded_data[0].size();
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < data_num_; ++i) {
                hnsw.addPoint(encoded_data[i].data(), i);
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

            {
                std::filesystem::path fsPath(codebooks_path_);
                fsPath.remove_filename();
                std::filesystem::create_directories(fsPath);
                std::ofstream out(codebooks_path_, std::ios::binary);
                auto& codebooks = hnswlib::codebooks_;
                codebooks.resize(subvector_num_);
                for (int i = 0; i < subvector_num_; ++i) {
                    codebooks[i].resize(CLUSTER_NUM);
                    for (int j = 0; j < CLUSTER_NUM; ++j) {
                        codebooks[i][j].resize(SUBVECTOR_LENGTH);
                        for (int k = 0; k < SUBVECTOR_LENGTH; ++k) {
                            out.write(reinterpret_cast<char*>(&codebooks[i][j][k]), sizeof(float));
                        }
                    }
                }
                auto& dist = hnswlib::dist_;
                dist.resize(subvector_num_);
                for (int i = 0; i < subvector_num_; ++i) {
                    dist[i].resize(CLUSTER_NUM);
                    for (int j = 0; j < CLUSTER_NUM; ++j) {
                        dist[i][j].resize(CLUSTER_NUM);
                        for (int k = 0; k < CLUSTER_NUM; ++k) {
                            out.write(reinterpret_cast<char*>(&dist[i][j][k]), sizeof(float));
                        }
                    }
                }
                hnsw.saveIndex(index_path_);
            }
        }

        // Encode query using PQ
        auto s_solve = std::chrono::system_clock::now();
        std::vector<std::vector<data_t>> encoded_query(query_num_);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < query_num_; ++i) {
            encoded_query[i] = pqEncode(query_set_[i]);
        }
        hnsw.setEf(ef_search_);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < query_num_; ++i) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hnsw.searchKnn(encoded_query[i].data(), K);
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

protected:
    void generate_codebooks(size_t sample_num) {
        std::vector<int> data(sample_num);
        std::iota(data.begin(), data.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(data.begin(), data.end(), g);
        std::vector<int> subset_data(data.begin(), data.begin() + sample_num);

        auto& codebooks = hnswlib::codebooks_;
        codebooks.resize(subvector_num_);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < subvector_num_; ++i) {
            MatrixXf subvector_data(sample_num, subvector_length_);
            for (size_t j = 0; j < sample_num; ++j) {
                subvector_data.row(j) = Eigen::Map<VectorXf>(data_set_[subset_data[j]].data() + i * subvector_length_, subvector_length_);
            }
            MatrixXf centroid_matrix = kMeans(subvector_data, cluster_num_, max_iterations_);

            // Change Eigen vectors to std::vector
            std::vector<std::vector<float>> centroids(centroid_matrix.rows(), std::vector<float>(centroid_matrix.cols()));
            for (int r = 0; r < centroid_matrix.rows(); ++r) {
                Eigen::VectorXf row = centroid_matrix.row(r);
                std::copy(row.data(), row.data() + row.size(), centroids[r].begin());
            }
            codebooks[i] = centroids;
        }

        auto& dist = hnswlib::dist_;
        dist = std::vector<std::vector<std::vector<float>>>(
            subvector_num_,
            std::vector<std::vector<float>>(
                cluster_num_,
                std::vector<float>(cluster_num_, 0.0f)
            )
        );
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < subvector_num_; ++i) {
            for (size_t j1 = 0; j1 < cluster_num_; ++j1) {
                for (size_t j2 = j1 + 1; j2 < cluster_num_; ++j2) {
                    VectorXf v1 = Eigen::Map<VectorXf>(codebooks[i][j1].data(), subvector_length_);
                    VectorXf v2 = Eigen::Map<VectorXf>(codebooks[i][j2].data(), subvector_length_);
                    dist[i][j1][j2] = (v1 - v2).squaredNorm();
                    // dist[i][j2][j1] = dist[i][j1][j2];
                }
            }
        }
    }

    MatrixXf kMeans(const MatrixXf& data_set, size_t cluster_num, size_t max_iterations) {
        // Initialize centroids randomly
        size_t data_num = data_set.rows();
        size_t data_dim = data_set.cols();
        MatrixXf centroids(cluster_num, data_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data_num - 1);
        for (size_t i = 0; i < cluster_num; ++i) {
            centroids.row(i) = data_set.row(dis(gen));
        }

        // kMeans
        std::vector<size_t> labels(data_num);
        auto startTime = std::chrono::steady_clock::now();
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            progressBar(iter, max_iterations, startTime);

            // Assign labels based on closest centroid
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < data_num; ++i) {
                float min_dist = FLT_MAX;
                size_t best_index = 0;
                for (size_t j = 0; j < cluster_num; ++j) {
                    float dist = (data_set.row(i) - centroids.row(j)).squaredNorm();
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_index = j;
                    }
                }
                labels[i] = best_index;
            }

            // Update centroids
            MatrixXf new_centroids = MatrixXf::Zero(cluster_num, data_dim);
            std::vector<int> counts(cluster_num, 0);
            for (size_t i = 0; i < data_num; ++i) {
                new_centroids.row(labels[i]) += data_set.row(i);
                counts[labels[i]]++;
            }
            for (size_t j = 0; j < cluster_num; ++j) {
                if (counts[j] != 0) {
                    new_centroids.row(j) /= counts[j];
                } else {
                    new_centroids.row(j) = data_set.row(dis(gen)); // Reinitialize a random centroid if no points are assigned
                }
            }
            centroids = new_centroids;
        }
        progressBar(max_iterations, max_iterations, startTime);
        return centroids;
    }

    std::vector<data_t> pqEncode(const std::vector<float>& vec) {
        std::vector<data_t> encoded_vector(subvector_num_);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < subvector_num_; ++i) {
            float min_dist = FLT_MAX;
            data_t best_index = 0;
            for (size_t j = 0; j < cluster_num_; ++j) {
                float dist = 0.0f;
                for (size_t k = 0; k < subvector_length_; ++k) {
                    float diff = vec[i * subvector_length_ + k] - hnswlib::codebooks_[i][j][k];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_index = j;
                }
            }
            encoded_vector[i] = best_index;
        }
        return encoded_vector;
    }

protected:
    size_t subvector_num_;
    size_t subvector_length_;
    size_t cluster_num_;
    size_t max_iterations_;
    size_t sample_num_;
};
