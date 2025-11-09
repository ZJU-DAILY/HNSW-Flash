#include "flash_strategy.h"

int
main(int argc, char **argv) {
    std::string dataset        = argv[1];
    std::string solve_strategy = argv[2];

    // Initialization
    std::string source_path;
    std::string query_path;
    std::string gt_path;
    std::string knn_path;
    std::string codebooks_path;
    std::string index_path;

    // Create a filename for saving the index
    std::string suffix  = solve_strategy + "_";
    suffix             += std::to_string(EF_CONSTRUCTION) + "_";
    suffix             += std::to_string(M) + "_";
    suffix             += std::to_string(SUBVECTOR_NUM) + "_";
    suffix             += std::to_string(PRINCIPAL_DIM) + "_";

    source_path         = "../data/" + dataset + "/" + dataset + "_base.fvecs";
    query_path          = "../data/" + dataset + "/" + dataset + "_query.fvecs";
    gt_path             = "../data/" + dataset + "/" + dataset + "_groundtruth.ivecs";
    knn_path            = "../statistics/knns/" + dataset + "_knn.ivecs";
    codebooks_path      = "../statistics/codebooks/" + dataset + "/codebooks_" + suffix;
    index_path          = "../statistics/codebooks/" + dataset + "/index_" + suffix;

    SolveStrategy *strategy;
    if (solve_strategy == "flash") {
        strategy = new FlashStrategy(source_path, query_path, codebooks_path, index_path);
    } else {
        std::cout << "Unknown strategy: " << strategy << std::endl;
        std::cout << "['flash']" << std::endl;
        return 1;
    }

    // Processing
    strategy->solve();
    // strategy->save_knn(knn_path);
    strategy->recall(gt_path);
    return 0;
}
