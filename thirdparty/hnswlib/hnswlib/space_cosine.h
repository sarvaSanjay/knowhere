#pragma once

#include "hnswlib.h"
#include "simd/hook.h"

namespace hnswlib {

template <typename DataType, typename DistanceType>
static DistanceType
Cosine(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    std::cout << "Asamwalaykum lyiari" << std::endl;
    size_t dim = *((size_t*)qty_ptr);
    size_t half_dim = dim / 2;
    size_t second_half_dim = dim - half_dim;
    
    DistanceType first_half_product = 0.0;
    DistanceType second_half_product = 0.0;
    
    if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
        first_half_product = faiss::fvec_inner_product((const DataType*)pVect1, (const DataType*)pVect2, half_dim);
        second_half_product = faiss::fvec_inner_product(
            (const DataType*)pVect1 + half_dim, 
            (const DataType*)pVect2 + half_dim, 
            second_half_dim);
    } else if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
        first_half_product = faiss::fp16_vec_inner_product((const DataType*)pVect1, (const DataType*)pVect2, half_dim);
        second_half_product = faiss::fp16_vec_inner_product(
            (const DataType*)pVect1 + half_dim, 
            (const DataType*)pVect2 + half_dim, 
            second_half_dim);
    } else if constexpr (std::is_same_v<DataType, knowhere::bf16>) {
        first_half_product = faiss::bf16_vec_inner_product((const DataType*)pVect1, (const DataType*)pVect2, half_dim);
        second_half_product = faiss::bf16_vec_inner_product(
            (const DataType*)pVect1 + half_dim, 
            (const DataType*)pVect2 + half_dim, 
            second_half_dim);
    } else {
        throw std::runtime_error("Unknown Datatype\n");
    }
    
    return 2.0 * first_half_product + 0.5 * second_half_product;
}

template <typename DataType, typename DistanceType>
static DistanceType
CosineDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * Cosine<DataType, DistanceType>(pVect1, pVect2, qty_ptr);
}

static inline float
CosineSQ8Distance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * faiss::ivec_inner_product((const int8_t*)pVect1, (const int8_t*)pVect2, *(size_t*)qty_ptr);
}

template <typename DataType, typename DistanceType>
class CosineSpace : public SpaceInterface<DistanceType> {
    DISTFUNC<DistanceType> fstdistfunc_;
    DISTFUNC<float> fstdistfunc_sq_;
    size_t data_size_;
    size_t dim_;

 public:
    CosineSpace(size_t dim) {
        fstdistfunc_ = CosineDistance<DataType, DistanceType>;
        fstdistfunc_sq_ = CosineSQ8Distance;
        dim_ = dim;
        data_size_ = dim * sizeof(DataType);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<DistanceType>
    get_dist_func() {
        return fstdistfunc_;
    }

    DISTFUNC<float>
    get_dist_func_sq() {
        return fstdistfunc_sq_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~CosineSpace() {
    }
};

}  // namespace hnswlib
