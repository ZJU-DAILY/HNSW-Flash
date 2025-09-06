# HNSW-Flash

This repository contains the code for the paper **Accelerating Graph Indexing for ANNS on Modern CPUs**.

This code builds upon [hnswlib](https://github.com/nmslib/hnswlib) and [Eigen](https://gitlab.com/libeigen/eigen).

## Prerequisite

### Datasets

**File format:** The data used in this project is formatted in `.fvecs` and  `.ivecs`. Each file contains multiple lines of vectors; each line begins with an integer $d$, representing the dimension of the vector, followed by $d$ numbers as vector data. The number are of type `float` in `.fvecs` files and `integer` in `.ivecs` files. For more information, visit [this site](http://corpus-texmex.irisa.fr/).

Below are the datasets mentioned in the paper:

| Datasets                                                                                                    | Data Volume   | Size (GB) | Dim.  | Query Volume |
| ----------------------------------------------------------------------------------------------------------- | ------------- | --------- | ----- | ------------ |
| cohere10k (for test) [(link)](data/cohere10k)                                                    | 10,000     | 0.029     | 768   | 100       |
| ARGILLA [(link)](https://huggingface.co/datasets/argilla-warehouse/personahub-fineweb-edu-4-embeddings)     | 21,071,228    | 81        | 1,024 | 100,000      |
| ANTON [(link)](https://huggingface.co/datasets/anton-l/wiki-embed-mxbai-embed-large-v1)                     | 19,399,177    | 75        | 1,024 | 100,000      |
| LAION [(link)](https://sisap-challenges.github.io/2024/datasets/)                                           | 100,000,000   | 293       | 768   | 100,000      |
| IMAGENET [(link)](<https://huggingface.co/datasets/kinianlo/ imagenet_embeddings.>)                         | 13,158,856    | 38        | 768   | 100,000      |
| COHERE [(link)](https://huggingface.co/datasets/Cohere/wikipedia-22-12-es-embeddings)                       | 10,124,929    | 30        | 768   | 100,000      |
| DATACOMP [(link)](https://huggingface.co/datasets/nielsr/datacomp-small-with-embeddings-and-cluster-labels) | 12,800,000    | 37        | 768   | 100,000      |
| BIGCODE [(link)](https://huggingface.co/datasets/bigcode/stack-exchange-embeddings-20230914)                | 10,404,628    | 30        | 768   | 100,000      |
| SSNPP [(link)](https://big-ann-benchmarks.com/neurips21.html)                                               | 1,000,000,000 | 960       | 256   | 100,000      |

The following is an example of the file organization for the `cohere10k` dataset:

    HNSW_Flash
    ├─data
    │  ├─cohere10k
    │  │  ├─cohere10k_base.fvecs
    │  │  ├─cohere10k_query.fvecs
    │  │  └─cohere10k_groundtruth.ivecs
    │  ├─...

### Strategies

Below are the strategies implemented in this repository. The strategy names are used to select different approaches when running the code.

| **Strategy** | **Description** |  
|--------------|-----------------|  
| **flash**    | A high-performance implementation of HNSW, optimized for fast search in large-scale datasets. Utilizes techniques specific to **Flash** for efficient distance calculations. |  
| **hnsw**     | A standard implementation of **Hierarchical Navigable Small World (HNSW)** graph, based on the version in [nmslib/hnswlib]((https://github.com/nmslib/hnswlib)). This is the core algorithm for approximate nearest neighbor search. | 
| **nsg**    | An implementation of **Navigating Spreading-out Graph (NSG)**. |  
| **nsg-flash**     | Base on NSG algorithm, utilizes the same techniques as **Flash** for efficient distance calculations. |  
| **pca-sdc**  | Extends HNSW by applying **Principal Component Analysis (PCA)** for dimensionality reduction. Uses **SDC** (Simultaneous Distance Calculation) for efficient query processing, improving both speed and accuracy. |  
| **pq-adc**   | Combines HNSW with **Product Quantization (PQ)**. Uses **ADC** (Asymmetric Distance Calculation) to perform queries efficiently by comparing encoded vectors with raw data. Suitable for scenarios where memory usage is a concern. |  
| **pq-sdc**   | Similar to `pq-adc`, but uses **SDC** for query processing, balancing speed and accuracy. Ideal for large datasets where encoding and fast computation are critical. |  
| **sq-adc**   | Integrates HNSW with **Scalar Quantization (SQ)**. Uses **ADC** for distance calculations, enabling fast search with compressed data. This method is useful for low-memory environments. |  
| **sq-sdc**   | A variation of `sq-adc`, utilizing **SDC** for querying. Offers a trade-off between accuracy and efficiency, particularly useful for datasets with high dimensionality. |
| **taumg**    | An implementation of **τ-Monotonic Graph (τ-MG)**. |  
| **taumg-flash**     | Base on τ-MG algorithm, utilizes the same techniques as **Flash** for efficient distance calculations. |


### Environment

-   cmake >= 3.15
-   g++ >= 11.4 with C++17 support
-   CPU with SSE/AVX/AVX2/AVX512 support

## Reproduction

The algorithms automatically check if the index has been generated and saved in the file system. if not, they will generate and save the index before running the query; otherwise, they will load the existing index file and run the query.

The `cohere10k` dataset has been pre-configured to run with all strategies:

    make build
    make run flash cohere10k

For other datasets, you can run the code using the following format:

    cd ./bin && ./main [dataset_name] [strategy_name]

For more details on running the code, please refer to `./main.cc` and `./Makefile`.

All parameters for the strategies can be modified in `./include/core.h`. Please recompile the program after making any changes.
