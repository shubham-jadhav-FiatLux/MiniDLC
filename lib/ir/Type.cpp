#include "dlc/ir/Type.h"
#include <sstream>

namespace dlc {

std::string dataTypeToString(DataType type) {
    switch (type) {
        case DataType::Float32: return "f32";
        case DataType::Int32:   return "i32";
    }
    return "unknown";
}

bool computeBroadcastShape(const Shape& a, const Shape& b, Shape& outShape) {
    size_t rankA = a.size();
    size_t rankB = b.size();
    size_t maxRank = std::max(rankA, rankB);
    outShape.resize(maxRank);

    for (size_t i = 0; i < maxRank; ++i) {
        int64_t dimA = (i < rankA) ? a[rankA - 1 - i] : 1;
        int64_t dimB = (i < rankB) ? b[rankB - 1 - i] : 1;

        if (dimA == dimB) {
            outShape[maxRank - 1 - i] = dimA;
        } else if (dimA == 1) {
            outShape[maxRank - 1 - i] = dimB;
        } else if (dimB == 1) {
            outShape[maxRank - 1 - i] = dimA;
        } else {
            return false; // Incompatible shapes
        }
    }
    return true;
}

int64_t TensorType::getNumElements() const {
    if (shape_.empty()) {
        return 1; // 0-D scalar has 1 element
    }
    int64_t count = 1;
    for (int64_t dim : shape_) {
        count *= dim;
    }
    return count;
}

int64_t TensorType::getSizeBytes() const {
    int64_t bytesPerElem = (elemType_ == DataType::Float32) ? 4 : 4;
    return getNumElements() * bytesPerElem;
}

std::string TensorType::toString() const {
    std::ostringstream oss;
    oss << dataTypeToString(elemType_);
    oss << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ",";
        oss << shape_[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace dlc
