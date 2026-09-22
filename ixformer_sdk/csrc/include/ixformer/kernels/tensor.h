#pragma once

#include <string>

namespace ixformer::kernels {

const uint8_t MAX_TENSOR_NDIM = 8;

// align with at::ScalarType
enum DType {
    Byte = 0,
    Char = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Half = 5,
    Float = 6,
    Double = 7,
    ComplexHalf = 8,
    ComplexFloat = 9,
    ComplexDoubl = 10,
    Bool = 11,
    QInt8 = 12,
    QUInt8 = 13,
    QInt32 = 14,
    BFloat16 = 15,
    QUInt4x2 = 16,
    QUInt2x4 = 17,
    Bits1x8 = 18,
    Bits2x4 = 19,
    Bits4x2 = 20,
    Bits8 = 21,
    Bits16 = 22,
    Float8_e5m2 = 23,
    Float8_e4m3fn = 24,
    Undefined = 25,
    NumOptions = 26
};

struct TensorDesc {

public:
    // delete default constructor
    TensorDesc() = delete;
    // All information must be (should be) prepared when constructing a TensorDesc object.
    TensorDesc(DType scalar_type, void *data_ptr, int64_t numel, int64_t dim, const int64_t *size, const int64_t *stride, bool is_contiguous, bool is_cuda)
        : dtype(scalar_type), ptr(data_ptr), nnumel(numel), ndim(dim), sizes(size), strides(stride), contiguous(is_contiguous), cuda(is_cuda) {}

    inline DType scalar_type() const {
        return dtype;
    }

    inline void *data_ptr() const {
        return ptr;
    }

    inline int64_t numel() const {
        return nnumel;
    }

    inline int64_t dim() const {
        return ndim;
    }

    inline int64_t size(int64_t dim) const {
        return dim < 0 ? sizes[ndim - dim] : sizes[dim];
    }

    inline int64_t stride(int64_t dim) const {
        return dim < 0 ? strides[ndim - dim] : strides[dim];
    }

    inline bool is_contiguous() const {
        return contiguous;
    }

    inline bool is_cuda() const {
        return cuda;
    }

private:
    void *ptr{nullptr};
    DType dtype;
    int64_t nnumel{0};
    int64_t ndim{0};
    const int64_t *sizes{nullptr};
    const int64_t *strides{nullptr};
    bool contiguous{false};
    bool cuda{false};
};

}// namespace ixformer::kernels
