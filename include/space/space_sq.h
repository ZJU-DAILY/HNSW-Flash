#pragma once
#include "../third_party/hnswlib/hnswlib.h"

namespace hnswlib {

float max_value_;
float min_value_;            // offset
float max_minus_min_value_;
float alpha_;

static std::vector<data_t> 
sqEncode(const std::vector<float>& vec) {
    size_t dim = vec.size();
    std::vector<data_t> encoded_vector(dim);
    for (size_t j = 0; j < dim; ++j) {
        float value = (vec[j] - min_value_) / max_minus_min_value_;
        if (value < 0) value = 0;
        if (value > 1) value = 1;
        encoded_vector[j] = (data_t)((double)std::numeric_limits<data_t>::max() * value);
    }
    return encoded_vector;
}

static std::vector<float> 
sqDecode(const data_t* vec, size_t dim) {
    std::vector<float> decoded_vector(dim);
    for (size_t j = 0; j < dim; ++j) {
        float value = min_value_ + (vec[j] + 0.5) * alpha_;
        decoded_vector[j] = value;
    }
    return decoded_vector;
} 

static float
SqSdcL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    data_t *pVect1 = (data_t *) pVect1v;
    data_t *pVect2 = (data_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t j = 0; j < qty; ++j) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

static float 
SqAdcL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    data_t *pVect2 = (data_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    
    std::vector<float> vec2 = sqDecode(pVect2, qty);

    float res = 0;
    for (size_t j = 0; j < qty; ++j) {
        float t = *pVect1 - vec2[j];
        pVect1++;
        res += t * t;
    }
    return (res);
}

class SqSdcSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

    void train_quantizer(const std::vector<std::vector<float>>& data_set) {
        size_t data_num = data_set.size();
        size_t data_dim = data_set[0].size();
        max_value_ = std::numeric_limits<float>::min();
        min_value_ = std::numeric_limits<float>::max();
        for (size_t i = 0; i < data_num; ++i) {
            for (size_t j = 0; j < data_dim; ++j) {
                min_value_ = std::min(min_value_, data_set[i][j]);
                max_value_ = std::max(max_value_, data_set[i][j]);
            }
        }

        alpha_ = (double)(max_value_ - min_value_) / std::numeric_limits<data_t>::max();
        max_minus_min_value_ = max_value_ - min_value_;
    }

public:
    SqSdcSpace(size_t dim, const std::vector<std::vector<float>>& data_set) {
        fstdistfunc_ = SqSdcL2Sqr;
        dim_ = dim;
        data_size_ = dim * sizeof(data_t);
        encode_ = sqEncode;

        train_quantizer(data_set);
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

    using EncoderFunction = std::vector<data_t>(*)(const std::vector<float>&);
    EncoderFunction encode_;
};

}  // namespace hnswlib
