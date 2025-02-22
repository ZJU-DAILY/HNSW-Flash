# Space  

Functions for calculating distances.

### `space_flash.h`  
Used in **Flash**.  
Uses `FlashL2Sqr` to compute distances.

### `space_pq.h`  
Used in **PQ-ADC** and **PQ-SDC**.  
Uses `PqSdcL2Sqr` and `PqAdcL2Sqr` to compute distances.

### `space_sq.h`  
Used in **SQ-ADC** and **SQ-SDC**.  
- Uses `SqSdcL2Sqr` and `SqAdcL2Sqr` for distance computation.  
- Uses `SqEncode` and `SqDecode` for encoding and decoding data.  
- Automatically invokes `train_quantizer` to calculate encoding parameters when `SqSdcSpace` is declared.

### Other Algorithms  
**HNSW** and **PCA-SDC** use `third_party/hnswlib/l2_space.h`.  
Distance is computed using `L2Sqr`.
