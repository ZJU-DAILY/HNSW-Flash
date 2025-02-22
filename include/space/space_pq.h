#pragma once
#include "../third_party/hnswlib/hnswlib.h"

namespace hnswlib {

std::vector<std::vector<std::vector<float>>> codebooks_;
std::vector<std::vector<std::vector<float>>> dist_;

static float
PqSdcL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    data_t *pVect1 = (data_t *) pVect1v;
    data_t *pVect2 = (data_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
        data_t index1 = pVect1[i];
        data_t index2 = pVect2[i];
        if (index1 > index2) std::swap(index1, index2);
        res += dist_[i][index1][index2];
    }
    return (res);
}

static float
PqAdcL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    data_t *pVect2 = (data_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < SUBVECTOR_NUM; ++i) {
        data_t index2 = pVect2[i];
        const std::vector<float>& centroid2 = codebooks_[i][index2];
        for (size_t j = 0; j < SUBVECTOR_LENGTH; ++j) {
            float t = centroid2[j] - pVect1[i * SUBVECTOR_LENGTH + j];
            res += t * t;
        }
    }
    return (res);
}

class PqSdcSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    PqSdcSpace(size_t dim) {
        fstdistfunc_ = PqSdcL2Sqr;
        dim_ = dim;
        data_size_ = dim * sizeof(data_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }
};

class PqAdcSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    PqAdcSpace(size_t dim) {
        fstdistfunc_ = PqAdcL2Sqr;
        dim_ = dim;
        data_size_ = dim * sizeof(data_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }
};

}  // namespace hnswlib
