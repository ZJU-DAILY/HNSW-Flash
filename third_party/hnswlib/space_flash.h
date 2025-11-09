#pragma once

#include <cstdint>

#include "../../third_party/hnswlib/hnswlib.h"

typedef uint8_t dist_table_t;
typedef uint8_t encoded_data_t;

namespace hnswlib {

#define ls(x) ((x >> 4) & 0x0F)
#define rs(x) (x & 0x0F)

/**
 * Calculate the squared Euclidean distance between two vectors
 * @param pVect1v Pointer to a distance table. The distance table contains
 * CLUSTER_NUM(16) distances for each subvector.
 * @param pVect2v Pointer to encoded data. The encoded data contains the cluster
 * indices of two subvectors, with each index stored in the high 4 bits and low
 * 4 bits, respectively.
 * @param qty_ptr Pointer to the dimension of the vectors
 * @return The squared Euclidean distance between the two vectors
 */
static uint32_t
FlashL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    dist_table_t *pVect1   = (dist_table_t *)pVect1v;    // distance table
    encoded_data_t *pVect2 = (encoded_data_t *)pVect2v;  // encoded data
    size_t qty             = *((size_t *)qty_ptr);

    uint32_t res           = 0;
    for (size_t i = 0; i < qty; ++i) {
        res    += (uint32_t)pVect1[rs(pVect2[i])];  // 偶数低位
        pVect1 += 16;
        res    += (uint32_t)pVect1[ls(pVect2[i])];  // 奇数高位
        pVect1 += 16;
    }
    return (res);
}

class FlashSpace: public SpaceInterface<uint32_t> {
    DISTFUNC<uint32_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    FlashSpace(size_t dim) {
        fstdistfunc_ = FlashL2Sqr;
        dim_         = dim;
        data_size_   = dim * sizeof(encoded_data_t);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<uint32_t>
    get_dist_func() {
        return fstdistfunc_;
    }

    void *
    get_dist_func_param() {
        return &dim_;
    }
};

}  // namespace hnswlib
