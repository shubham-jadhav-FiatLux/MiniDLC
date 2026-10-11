#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <ostream>

namespace dlc {

enum class DataType {
    Float32,
    Int32
};

std::string dataTypeToString(DataType type);

using Shape = std::vector<int64_t>;

class Type {
public:
    virtual ~Type() = default;
    virtual std::string toString() const = 0;
    virtual bool isTensor() const { return false; }
};

class TensorType : public Type {
public:
    TensorType(DataType elemType, Shape shape)
        : elemType_(elemType), shape_(std::move(shape)) {}

    DataType getElementType() const { return elemType_; }
    const Shape& getShape() const { return shape_; }
    int64_t getRank() const { return static_cast<int64_t>(shape_.size()); }
    int64_t getNumElements() const;
    int64_t getSizeBytes() const;

    bool isTensor() const override { return true; }
    std::string toString() const override;

    bool operator==(const TensorType& other) const {
        return elemType_ == other.elemType_ && shape_ == other.shape_;
    }
    bool operator!=(const TensorType& other) const {
        return !(*this == other);
    }

private:
    DataType elemType_;
    Shape shape_;
};

using TypePtr = std::shared_ptr<Type>;
using TensorTypePtr = std::shared_ptr<TensorType>;

} // namespace dlc
