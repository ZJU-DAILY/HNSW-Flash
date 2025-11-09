#pragma once

#include "eigen/Eigen/Dense"
#include "hnswlib/hnswalg_flash.h"
#include "solve_strategy.h"

using Eigen::MatrixXf;
using Eigen::VectorXf;

class FlashStrategy: public SolveStrategy {
 public:
    FlashStrategy(std::string source_path,
                  std::string query_path,
                  std::string codebooks_path,
                  std::string index_path)
        : SolveStrategy(source_path, query_path, codebooks_path, index_path) {
        subvector_num_ = SUBVECTOR_NUM;
        cluster_num_   = CLUSTER_NUM;
        sample_num_    = std::min(SAMPLE_NUM, (size_t)data_num_);
        ori_dim_       = data_dim_;
        principal_dim_ = PRINCIPAL_DIM;
        byte_num_      = subvector_num_ / 2;
        qmax           = 0;
    }

    ~FlashStrategy() {}

    void
    solve() {
        // --- Step 1. Initialize FlashSpace and HNSW structure ---
        hnswlib::FlashSpace flash_space(byte_num_);
        hnswlib::HierarchicalNSWFlash<uint32_t> *hnsw;

        // --- Step 2. Prepare dataset matrix ---
        Eigen::MatrixXf data_set(data_num_, data_dim_);
#pragma omp parallel for
        for (int i = 0; i < data_num_; ++i) {
            // Map original raw vector into Eigen::VectorXf
            Eigen::Map<Eigen::VectorXf> vec(data_set_[i].data(), ori_dim_);
            data_set.row(i) = vec.transpose();
        }

        // --- Step 3. Apply PCA encoding ---
        auto t_pca_start = std::chrono::system_clock::now();
        generate_matrix(data_set);
        pcaEncode(data_set);
        data_dim_      = PRINCIPAL_DIM;
        auto t_pca_end = std::chrono::system_clock::now();
        std::cout << "PCA encode data cost: " << time_cost(t_pca_start, t_pca_end) << " (ms)\n";

        // --- Step 4. Generate PQ codebooks ---
        auto t_codebook_start = std::chrono::system_clock::now();
        generate_codebooks(data_set);
        auto t_codebook_end = std::chrono::system_clock::now();
        std::cout << "Generate codebooks cost: " << time_cost(t_codebook_start, t_codebook_end)
                  << " (ms)\n";

        // --- Step 5. Build HNSW index ---
        auto t_build_start = std::chrono::system_clock::now();
        hnsw               = new hnswlib::HierarchicalNSWFlash<uint32_t>(
            &flash_space, data_num_, M_, ef_construction_);

        // Each point is PQ-encoded and inserted into the HNSW index
#pragma omp parallel for
        for (size_t i = 0; i < data_num_; ++i) {
            std::vector<uint8_t> encoded_data(subvector_num_ * cluster_num_ * sizeof(dist_table_t) +
                                              byte_num_ * sizeof(encoded_data_t));

            // Encode data into PQ representation
            pqEncode(data_set.row(i),
                     encoded_data.data() + subvector_num_ * cluster_num_ * sizeof(dist_table_t),
                     encoded_data.data());

            hnsw->addPoint(encoded_data.data(), i);
        }
        auto t_build_end = std::chrono::system_clock::now();
        std::cout << "Build index cost: " << time_cost(t_build_start, t_build_end) << " (ms)\n";

        // --- Step 6. Prepare query set ---
        auto t_search_start = std::chrono::system_clock::now();
        Eigen::MatrixXf query_set(query_num_, data_dim_);

#pragma omp parallel for
        for (int i = 0; i < query_num_; ++i) {
            Eigen::Map<Eigen::VectorXf> vec(query_set_[i].data(), ori_dim_);
            query_set.row(i) = vec.transpose();
        }

        // Apply PCA transformation on queries
        pcaEncode(query_set);
        hnsw->setEf(EF_SEARCH);

        // --- Step 7. Perform nearest neighbor search ---
#pragma omp parallel for
        for (size_t i = 0; i < query_num_; ++i) {
            std::vector<uint8_t> encoded_query(subvector_num_ * cluster_num_ *
                                                   sizeof(dist_table_t) +
                                               byte_num_ * sizeof(encoded_data_t));

            // PQ-encode query vector
            pqEncode(query_set.row(i),
                     encoded_query.data() + subvector_num_ * cluster_num_ * sizeof(dist_table_t),
                     encoded_query.data());

            // --- Optional: Re-ranking with original vectors ---
#if defined(RERANK)
            // Retrieve large candidate pool
            std::priority_queue<std::pair<uint32_t, hnswlib::labeltype>> tmp =
                hnsw->searchKnn(encoded_query.data(), K * 10);

            // Min-heap for final top-K results
            std::priority_queue<std::pair<float, hnswlib::labeltype>,
                                std::vector<std::pair<float, hnswlib::labeltype>>,
                                std::greater<>>
                result;

            // Compute real L2 distance for re-ranking
            while (!tmp.empty()) {
                float dist = 0.0f;
                size_t id  = tmp.top().second;
                tmp.pop();

                for (int j = 0; j < ori_dim_; ++j) {
                    float diff  = org_data_set_[id][j] - org_query_set_[i][j];
                    dist       += diff * diff;
                }
                result.emplace(dist, id);
            }

#else
            // --- Default: Use PQ distance directly ---
            std::priority_queue<std::pair<uint32_t, hnswlib::labeltype>> result =
                hnsw->searchKnn(encoded_query.data(), K);
#endif

            // Store top-K results
            while (!result.empty() && knn_results_[i].size() < K) {
                knn_results_[i].emplace_back(result.top().second);
                result.pop();
            }

            // Fill empty slots if results < K
            while (knn_results_[i].size() < K) {
                knn_results_[i].emplace_back(-1);
            }
        }

        auto t_search_end = std::chrono::system_clock::now();
        std::cout << "Search cost: " << time_cost(t_search_start, t_search_end) << " (ms)"
                  << std::endl;
    }

 protected:
    void
    generate_codebooks(const Eigen::MatrixXf &data_set) {
        const size_t data_num = data_set.rows();
        const int sub_dim     = data_dim_ / subvector_num_;

        std::vector<size_t> indices(data_num);
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 g(19260817);
        std::shuffle(indices.begin(), indices.end(), g);

        codebooks = Eigen::MatrixXf(subvector_num_ * cluster_num_, sub_dim);

#pragma omp parallel for
        for (size_t i = 0; i < subvector_num_; ++i) {
            // sample a subset of data
            Eigen::MatrixXf subvector_data(sample_num_, sub_dim);
            for (size_t j = 0; j < sample_num_; ++j) {
                subvector_data.row(j) = data_set.row(indices[j]).segment(i * sub_dim, sub_dim);
            }

            // KMeans clustering
            Eigen::MatrixXf centroids = kMeans(subvector_data, cluster_num_, 12);

            // store centroids in codebooks
            codebooks.block(i * cluster_num_, 0, cluster_num_, sub_dim) = centroids;
        }

        // calculate the max distance between centroids
        qmax = 0.0f;
#pragma omp parallel for reduction(max : qmax)
        for (size_t i = 0; i < subvector_num_; ++i) {
            Eigen::MatrixXf centroids = codebooks.block(i * cluster_num_, 0, cluster_num_, sub_dim);

            // Eigen vectorized distance calculation: all pairs distance matrix
            Eigen::MatrixXf dists =
                (centroids * centroids.transpose()).diagonal().replicate(1, cluster_num_) +
                (centroids * centroids.transpose())
                    .diagonal()
                    .transpose()
                    .replicate(cluster_num_, 1) -
                2 * (centroids * centroids.transpose());
            float local_max = dists.maxCoeff();
            qmax            = std::max(qmax, local_max);
        }
        std::cout << "qmax: " << qmax << std::endl;
    }

    /**
     * Perform k-means clustering on the given dataset
     * @param data Pointer to the dataset
     * @param k Number of clusters
     * @param max_iterations Maximum number of iterations
     * @return Returns the cluster center matrix
     */
    MatrixXf
    kMeans(const MatrixXf &data, size_t k, size_t max_iters) {
        const size_t n = data.rows(), dim = data.cols();
        MatrixXf centroids(k, dim);

        // initialize centroids randomly
        // std::random_device rd;
        // std::mt19937 gen(rd());
        std::mt19937 gen(114514);
        std::uniform_int_distribution<> dis(0, n - 1);
        for (size_t i = 0; i < k; ++i) centroids.row(i) = data.row(dis(gen));

        std::vector<size_t> labels(n);

        for (size_t iter = 0; iter < max_iters; ++iter) {
            // assign labels of the nearest centroid
            for (size_t i = 0; i < n; ++i) {
                float best_dist = FLT_MAX;
                size_t best     = 0;
                for (size_t j = 0; j < k; ++j) {
                    float dist = (data.row(i) - centroids.row(j)).squaredNorm();
                    if (dist < best_dist) {
                        best_dist = dist;
                        best      = j;
                    }
                }
                labels[i] = best;
            }

            // refresh centroids
            MatrixXf new_centroids = MatrixXf::Zero(k, dim);
            std::vector<int> counts(k, 0);
            for (size_t i = 0; i < n; ++i) {
                new_centroids.row(labels[i]) += data.row(i);
                counts[labels[i]]++;
            }

            for (size_t j = 0; j < k; ++j)
                centroids.row(j) =
                    counts[j] ? (new_centroids.row(j) / counts[j]).eval() : data.row(dis(gen));
        }
        return centroids;
    }

    /**
     * Perform PQ encoding on the given data and compute the distance table
     * between the encoded vectors and the original data. Then, perform SQ
     * encoding on the distance table with an upper bound of the sum of the
     * maximum distance from each subvector. When encoding base data, the
     * distance table with qmin and qmax remains stable. When encoding query
     * data, the distance table with qmin and qmax needs to be recalculated.
     * @param data Pointer to the data to be encoded
     * @param encoded_vector Pointer to the encoded vector
     * @param dist_table Pointer to the distance table
     */
    void
    pqEncode(const Eigen::VectorXf &data, uint8_t *encoded_vector, void *dist_table_) {
        dist_table_t *dist_table = static_cast<dist_table_t *>(dist_table_);

        const size_t dim         = data.size();
        const size_t sub_dim     = dim / subvector_num_;

        const float alpha        = 255.0f / qmax;

        for (size_t i = 0; i < subvector_num_; ++i) {
            Eigen::VectorXf sub       = data.segment(i * sub_dim, sub_dim);
            Eigen::MatrixXf centroids = codebooks.block(i * cluster_num_, 0, cluster_num_, sub_dim);

            // calculate the distance between the subvector and each centroid
            Eigen::VectorXf dists = (centroids.rowwise() - sub.transpose()).rowwise().squaredNorm();

            // find the index of the minimum distance
            Eigen::Index best_index;
            dists.minCoeff(&best_index);

            // quantize the distance and store it in the distance table
            for (size_t j = 0; j < cluster_num_; ++j) {
                dist_table[i * cluster_num_ + j] = static_cast<dist_table_t>(
                    std::min(dists[j] * alpha, (float)std::numeric_limits<dist_table_t>::max()));
            }

            // write the encoding (every 2 subspaces are merged into 1 byte)
            if (i % 2 == 0) {
                encoded_vector[i / 2] = (encoded_vector[i / 2] & 0xF0) | (best_index & 0x0F);
            } else {
                encoded_vector[i / 2] = (encoded_vector[i / 2] & 0x0F) | ((best_index & 0x0F) << 4);
            }
        }
    }

    void
    generate_matrix(Eigen::MatrixXf &data_set) {
        size_t data_num = data_set.rows();
        size_t data_dim = data_set.cols();

        if (sample_num_ > data_num) {
            sample_num_ = data_num;
        }

        std::vector<size_t> indices(data_num);
        std::iota(indices.begin(), indices.end(), 0);

        std::random_device rd;
        std::mt19937 g(19260817);
        // std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        Eigen::MatrixXf data(sample_num_, data_dim);

        for (int i = 0; i < sample_num_; ++i) {
            size_t idx  = indices[i];
            data.row(i) = data_set.row(idx);
        }

        data_mean_ = data.colwise().mean();

        for (int i = 0; i < sample_num_; ++i) {
            data.row(i) -= data_mean_.transpose();
        }

        Eigen::MatrixXf covariance_matrix = (data.adjoint() * data) / float(sample_num_ - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(covariance_matrix);
        principal_components = eigensolver.eigenvectors();
        principal_components = principal_components.rowwise().reverse();
        principal_components.conservativeResize(Eigen::NoChange, principal_dim_);

        // The pca projection is just for decreasing the dimension of data
        // then use random orthogonal matrix to project data
        MatrixXf A = MatrixXf::Random(principal_dim_, principal_dim_);
        Eigen::HouseholderQR<MatrixXf> qr(A);
        Eigen::MatrixXf orthogonal_matrix_ =
            qr.householderQ() * MatrixXf::Identity(principal_dim_, principal_dim_);

        principal_components = principal_components * orthogonal_matrix_;
    }

    void
    pcaEncode(Eigen::MatrixXf &data) {
        size_t data_num = data.rows();
        size_t data_dim = data.cols();

        for (size_t i = 0; i < data_num; ++i) {
            data.row(i) -= data_mean_.transpose();
        }

        data = data * principal_components;
    }

 protected:
    size_t sample_num_;
    size_t byte_num_;

    size_t ori_dim_;  // The original dim of data before PCA
    size_t principal_dim_;
    float qmax;  // The max bounds of SQ

    size_t subvector_num_;
    size_t cluster_num_;

    // float *codebooks;
    // std::array<std::array<std::array<float, CLUSTER_NUM>, CLUSTER_NUM>, SUBVECTOR_NUM> codebooks;

    Eigen::MatrixXf codebooks;
    Eigen::VectorXf data_mean_;            // Mean of data
    Eigen::MatrixXf principal_components;  // Principal components
};
