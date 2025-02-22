#pragma once

#include "adsampling.h"

namespace hnswlib {
    
#define ls(x) ((x >> 4) & 0x0F)
#define rs(x) (x & 0x0F)

/**
 * Calculate the squared Euclidean distance between two vectors
 * @param pVect1v Pointer to a distance table. The distance table contains CLUSTER_NUM distances for each subvector.
 * @param pVect2v Pointer to encoded data. The encoded data contains the cluster indices of two subvectors, with each index stored in the high 4 bits and low 4 bits, respectively.
 * @param qty_ptr Pointer to the dimension of the vectors
 * @return The squared Euclidean distance between the two vectors
 */
static data_t
FlashL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    data_t *pVect1 = (data_t *) pVect1v;    // distance table
    uint8_t *pVect2 = (uint8_t *) pVect2v;  // encoded data
    size_t qty = *((size_t *) qty_ptr);

    data_t res = 0;
    if (CLUSTER_NUM == 16) {
        int tmp = qty / BATCH;
        for (int i = 0; i < tmp; ++i) {
            for (int j = 0; j < BATCH; ++j) {
                res += *(pVect1 + ls(*(pVect2 + j)));
                pVect1 += CLUSTER_NUM;
            }
            for (int j = 0; j < BATCH; ++j) {
                res += *(pVect1 + rs(*(pVect2 + j)));
                pVect1 += CLUSTER_NUM;
            }
            pVect2 += BATCH;
        }
    } else {
        int tmp = qty;
        for (int i = 0; i < tmp; ++i) {
            res += *(pVect1 + (*pVect2));
            pVect1 += CLUSTER_NUM;
            pVect2 ++;
        }
    }
    return (res);
}

class FlashSpace : public SpaceInterface<data_t> {
    DISTFUNC<data_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    FlashSpace(size_t dim) {
        fstdistfunc_ = FlashL2Sqr;
        dim_ = dim;
        data_size_ = dim * sizeof(data_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<data_t> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }
};

}  // namespace hnswlib
