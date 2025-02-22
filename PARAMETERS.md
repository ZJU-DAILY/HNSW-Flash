# Parameters

| **Parameter**         | **Description**                                              | **PQ** | **SQ** | **PCA** | **Flash** |
| --------------------- | ------------------------------------------------------------ | :----: | :----: | :-----: | :-------: |
| **K**                 | Specifies the top K nearest neighbors returned by the algorithm. |   √    |   √    |    √    |     √     |
| **M**                 | Number of neighbors in higher layers. The base layer contains twice as many neighbors as higher layers. <br /> **Must be a multiple of `VECTOR_PER_BLOCK` in Flash.** <br /> Corresponds to $R$ in the original paper. |   √    |   √    |    √    |     √     |
| **EF_CONSTRUCTION**   | Maximum number of candidate neighbors considered during index construction. The algorithm prunes these candidates to form the final graph structure. <br /> Corresponds to $C$ in the original paper. <br /> A higher value improves graph quality but increases indexing time. |   √    |   √    |    √    |     √     |
| **EF_SEARCH**         | Maximum number of candidates retained during the search phase. The algorithm selects and evaluates unvisited candidates iteratively. <br /> A higher value improves search quality but increases search time. |   √    |   √    |    √    |     √     |
| **SAMPLE_NUM**        | In PQ: Number of points sampled for k-means clustering. <br /> In PCA: Number of points used to extract principal components. |   √    |        |    √    |     √     |
| **NUM_THREADS**       | Number of threads used for parallel processing on the CPU.   |   √    |   √    |    √    |     √     |
| **SUBVECTOR_NUM**     | Number of subvectors created by splitting the original dimensions. <br /> Alternatively, this represents the dimensionality after applying PQ. <br /> **Must be a multiple of `VECTOR_PER_BLOCK` in Flash.** |   √    |        |         |     √     |
| **SUBVECTOR_LENGTH**  | Number of dimensions in each subvector. <br /> Total dimensions = `SUBVECTOR_NUM` * `SUBVECTOR_LENGTH`. <br /> **Deprecated.** |   √    |        |         |     √     |
| **CLUSTER_NUM**       | Number of clusters used in PQ for k-means clustering.        |   √    |        |         |     √     |
| **MAX_ITERATIONS**    | Maximum number of iterations for k-means clustering in PQ.   |   √    |        |         |     √     |
| **VECTORS_PER_BLOCK** | Number of neighbors Flash can process simultaneously. <br /> Relevant when `PQLINK_CALC` is enabled and the CPU supports SIMD. <br /> Typically 16 for SSE/AVX; may change to 64 with AVX512. |        |        |         |     √     |
| **PRINCIPAL_DIM**     | Number of dimensions retained after PCA transformation.      |        |        |    √    |     √     |

# Macro Definition

| **Parameter**                  | **Description**                                              |
| ------------------------------ | ------------------------------------------------------------ |
| **INT8/INT16/INT32 (General)** | Data type for `data_t`. <br /> **PQ** and **PCA** only support `INT32`. <br /> **Unsigned data types are used when `ALL_POSITIVE_NUMBER` is enabled.** |
| **ALL_POSITIVE_NUMBER**        | Modifies `nmslib/hnswlib/hnswalg.h` to ensure all numbers are positive, <br /> allowing the use of unsigned data types. |
| **PQLINK_STORE**               | Stores the neighbor vectors for each node.                   |
| **PQLINK_CALC**                | Calculates neighbor data in batches. <br /> **Must be used together with `PQLINK_STORE`.** |
| **USE_PCA**                    | Applies PCA to reduce dimensions in Flash.                   |
| **USE_PCA_OPTIMAL**            | Groups subvectors based on the sum of cumulative variance in PCA.<br /> **May slow down related algorithms.** |
| **RERANK**                     | Searches for $2K$ points and re-ranks them using the original distance metric. |
| **SAVE_MEMORY**                | Skips saving the distance table when using SDC to calculate distances. |
| **TAU**                        | Parameter used in tau-MG                                     |
| **VBASE**                      | use vbase to calculate distance                              |
| **VBASE_WINDOW_SIZE**          | the window size of vbase                                     |
| **ADSAMPLING**                 | use ADSampling to calculate distance                         |
| **ADSAMPLING_EPSILON**         | the epsilon of ADSampling                                    |
| **ADSAMPLING_DELTA_D**         | the delta_d of ADSampling                                    |

