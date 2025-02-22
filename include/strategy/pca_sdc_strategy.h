#pragma once

#include "solve_strategy.h"
#include "../../third_party/hnswlib/hnswlib.h"

class PcaSdcStrategy: public SolveStrategy {
public:
    PcaSdcStrategy(std::string source_path, std::string query_path, std::string codebooks_path, std::string index_path):
        SolveStrategy(source_path, query_path, codebooks_path, index_path) {
        subvector_num_ = SUBVECTOR_NUM;
        subvector_length_ = SUBVECTOR_LENGTH = data_dim_ / subvector_num_;
        sample_num_ = std::min(SAMPLE_NUM, (size_t)data_num_);
        ori_dim = data_dim_;
    }

    void solve() {
        hnswlib::L2Space l2space(PRINCIPAL_DIM);
        hnswlib::HierarchicalNSW<float> hnsw(&l2space, data_num_, M_, ef_construction_);

        if (std::filesystem::exists(codebooks_path_)) {
            std::ifstream in(codebooks_path_, std::ios::binary);
            std::vector<std::vector<float>> encoded_data;
            pcaEncode(data_set_, encoded_data);
            Eigen::VectorXf tmp1(data_dim_);
            for (int j = 0; j < data_dim_; ++j) {
                in.read(reinterpret_cast<char*>(&tmp1(j)), sizeof(double));
            }
            data_mean_ = tmp1;
            Eigen::MatrixXf tmp2(data_dim_, PRINCIPAL_DIM);
            for (int i = 0; i < data_dim_; ++i) {
                for (int j = 0; j < PRINCIPAL_DIM; ++j) {
                    in.read(reinterpret_cast<char*>(&tmp2(i, j)), sizeof(double));
                }
            }
            principal_components = tmp2;
            data_dim_ = PRINCIPAL_DIM;

            hnsw.loadIndex(index_path_, &l2space, data_num_);
        } else {
            // Generate/Read codebooks
            auto s_gen = std::chrono::system_clock::now();
            generate_matrix(PRINCIPAL_DIM, sample_num_);
            // generate_matrix(0.01);
            std::cout << "dim: " << data_dim_  << std::endl;
            auto e_gen = std::chrono::system_clock::now();
            std::cout << "generate codebooks cost: " << time_cost(s_gen, e_gen) << " (ms)\n";

            // Encode data using PQ
            auto s_encode_data = std::chrono::system_clock::now();
            std::vector<std::vector<float>> encoded_data;
            pcaEncode(data_set_, encoded_data);
            auto e_encode_data = std::chrono::system_clock::now();
            std::cout << "encode data cost: " << time_cost(s_encode_data, e_encode_data) << " (ms)\n";

            // Build HNSW index
            auto s_build = std::chrono::system_clock::now();
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < data_num_; ++i) {
                hnsw.addPoint(encoded_data[i].data(), i);
            }
            auto e_build = std::chrono::system_clock::now();
            std::cout << "build cost: " << time_cost(s_build, e_build) << " (ms)\n";

            {
                std::filesystem::path fsPath(codebooks_path_);
                fsPath.remove_filename();
                std::filesystem::create_directories(fsPath);
                std::ofstream out(codebooks_path_, std::ios::binary);
                for (int j = 0; j < ori_dim; ++j) {
                    out.write(reinterpret_cast<char*>(&data_mean_(j)), sizeof(double));
                }
                for (int i = 0; i < ori_dim; ++i) {
                    for (int j = 0; j < PRINCIPAL_DIM; ++j) {
                        out.write(reinterpret_cast<char*>(&principal_components(i, j)), sizeof(double));
                    }
                }
                hnsw.saveIndex(index_path_);
            }
        }

        // Solve query
        auto s_solve = std::chrono::system_clock::now();
        std::vector<std::vector<float>> encoded_query;
        pcaEncode(query_set_, encoded_query);
        hnsw.setEf(ef_search_);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < query_num_; ++i) {
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
    }

protected:
    void generate_matrix(size_t new_dim, size_t sample_num) {
        std::vector<size_t> subset_data(data_set_.size());
        std::iota(subset_data.begin(), subset_data.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        // std::mt19937 g(19260817);
        std::shuffle(subset_data.begin(), subset_data.end(), g);

        size_t data_num = sample_num;
        size_t data_dim = data_set_[0].size();
        Eigen::MatrixXf data(sample_num, data_dim);
        for (int i = 0; i < data_num; ++i) {
            for (int j = 0; j < data_dim; ++j) {
                data(i, j) = data_set_[subset_data[i]][j];
            }
        }

        data_mean_ = data.colwise().mean();
        Eigen::MatrixXf data_centered = data.rowwise() - data_mean_.transpose();
        Eigen::MatrixXf covariance_matrix = (data_centered.adjoint() * data_centered) / float(data_num_ - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(covariance_matrix);
        Eigen::VectorXf eigenvalues = eigensolver.eigenvalues();
        Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors();
        Eigen::MatrixXf selected_eigenvectors = eigenvectors.rightCols(new_dim);

        data_dim_ = new_dim;
        principal_components = selected_eigenvectors;
    }

    // cumulative variance explained ratio
    void generate_matrix(double variance_threshold, size_t sample_num) {
        std::vector<size_t> subset_data(data_set_.size());
        std::iota(subset_data.begin(), subset_data.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        // std::mt19937 g(19260817);
        std::shuffle(subset_data.begin(), subset_data.end(), g);

        size_t data_num = sample_num;
        size_t data_dim = data_set_[0].size();
        Eigen::MatrixXf data(sample_num, data_dim);
        for (int i = 0; i < data_num; ++i) {
            for (int j = 0; j < data_dim; ++j) {
                data(i, j) = data_set_[subset_data[i]][j];
            }
        }

        data_mean_ = data.colwise().mean();
        Eigen::MatrixXf data_centered = data.rowwise() - data_mean_.transpose();
        Eigen::MatrixXf covariance_matrix = (data_centered.adjoint() * data_centered) / float(data_num_ - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(covariance_matrix);
        Eigen::VectorXf eigenvalues = eigensolver.eigenvalues();
        Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors();
        Eigen::VectorXf explained_variance_ratio = eigenvalues / eigenvalues.sum();

        Eigen::VectorXf cumulative_variance = explained_variance_ratio;
        for (int i = 1; i < cumulative_variance.size(); ++i) {
            cumulative_variance[i] += cumulative_variance[i - 1];
        }
        int new_dim = (cumulative_variance.array() < variance_threshold).count() + 1;

        Eigen::MatrixXf selected_eigenvectors = eigenvectors.rightCols(new_dim);

        data_dim_ = new_dim;
        principal_components = selected_eigenvectors;
    }

    void pcaEncode(const std::vector<std::vector<float>>& data_set, std::vector<std::vector<float>>& encoded_data) {
        size_t data_num = data_set.size();
        size_t original_dim = principal_components.rows();
        size_t new_dim = principal_components.cols();

        Eigen::MatrixXf data(data_num, original_dim);
        for (int i = 0; i < data_num; ++i) {
            for (int j = 0; j < original_dim; ++j) {
                data(i, j) = data_set[i][j];
            }
        }

        Eigen::MatrixXf data_centered = data.rowwise() - data_mean_.transpose();
        Eigen::MatrixXf data_pca = data_centered * principal_components;

        encoded_data.resize(data_num);
        for (int i = 0; i < data_num; ++i) {
            encoded_data[i].resize(new_dim);
            for (int j = 0; j < new_dim; ++j) {
                encoded_data[i][j] = data_pca(i, j);
            }
        }
    }

protected:
    size_t subvector_num_;
    size_t sample_num_;
    size_t subvector_length_;
    size_t ori_dim;

    Eigen::VectorXf data_mean_;
    Eigen::MatrixXf principal_components;
};
