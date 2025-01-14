#pragma once

#include <cassert>
#include <cmath>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

#include "puffinn/format/generic.hpp"
#include "puffinn/typedefs.hpp"
#include "puffinn/math.hpp"

namespace puffinn {
    // Bytes in a 256-bit vector
    const int VECTOR256_ALIGNMENT = 256/8;

    /// A format for storing real vectors of unit length.
    /// 
    /// Currently, only ``std::vector<float>``` is supported as input type. 
    /// The vectors do not need to be normalized before insertion.
    ///
    /// Each number is stored using a 16-bit fixed point format. 
    /// Although this slightly reduces the precision,
    /// the inaccuracies are very unlikely to have an impact on the result.
    /// The vectors are stored using 256-bit alignment.
    struct UnitVectorFormat {
        // Represent the values as signed 15bit fixed point numbers between -1 and 1.
        // Done since the values are always in the range [-1, 1].
        // This is equivalent to what is used by `mulhrs`. However this cannot represent 1 exactly.
        using Type = int16_t;
        /// Number of dimensions.
        using Args = unsigned int;

        const static unsigned int ALIGNMENT = VECTOR256_ALIGNMENT;

        // Number of `Type` values that fit into a 256 bit vector.
        const static unsigned int VALUES_PER_VEC = 16;

        // Convert a floating point value between -1 and 1 to the internal, fixed point representation.
        static constexpr int16_t to_16bit_fixed_point(float val) {
            assert(val >= -1.0 && val <= 1.0);

            val = std::min(val*(1 << 15), static_cast<float>(INT16_MAX));
            return static_cast<int16_t>(val);
        }

        // Convert a number between -1 and 1 from the internal,
        // fixed point representation to floating point.
        static constexpr float from_16bit_fixed_point(Type val) {
            return static_cast<float>(val)/(1 << 15);
        }

        static unsigned int storage_dimensions(Args dimensions) {
            return dimensions;
        }


        static uint64_t inner_memory_usage(Type&) {
            return 0;
        }

        static void store(
            const std::vector<float>& input,
            Type* storage,
            DatasetDescription<UnitVectorFormat> dataset
        ) {
            if (input.size() != dataset.args) {
                throw std::invalid_argument("input.size()");
            }

            std::vector<float> copy = input;
            float len_squared = 0.0;
            for (auto v : copy) {
                len_squared += v*v;
            }

            auto len = std::sqrt(len_squared);
            if (len != 0.0) {
                for (auto& v : copy) {
                    v /= len;
                }
            }

            for (size_t i=0; i < copy.size(); i++) {
                storage[i] = to_16bit_fixed_point(copy[i]);
            }
            for (size_t i=copy.size(); i < dataset.storage_len; i++) {
                storage[i] = to_16bit_fixed_point(0.0);
            }
        }

        static float innerProduct(int16_t* lhs, int16_t* rhs, unsigned int dimensions) {
            float dot = from_16bit_fixed_point(
                dot_product_i16_simple(lhs, rhs, dimensions));
            return dot; // Ensure the similarity is between 0 and 1.
        }

        static float innerProduct_avx(int16_t* lhs, int16_t* rhs, unsigned int dimensions) {
            float dot = from_16bit_fixed_point(
                dot_product_i16(lhs, rhs, dimensions));
            return dot; // Ensure the similarity is between 0 and 1.
        }
         
        

        static float distance(Type* p1, Type* p2, unsigned int dimensions) {
            float dist = 0;
            for(unsigned int i = 0; i < dimensions; i++) {
                float a = from_16bit_fixed_point(p1[i]),
                      b = from_16bit_fixed_point(p2[i]);
                dist += std::pow(a-b,2);
            }
            // the actual distance is unimportant so no need to take the square root 
            // dist = std::sqrt(dist)
            return dist;
        }
        static void add_assign_float(float* p1, Type* p2, unsigned int dimensions){
            for (unsigned int i = 0; i < dimensions; i++) {
                p1[i] += from_16bit_fixed_point(p2[i]);
            }
        }

        static void subtract_assign_float(float* p1, Type* p2, unsigned int dimensions){
            for (unsigned int i = 0; i < dimensions; i++) {
                p1[i] -= from_16bit_fixed_point(p2[i]);
            }
        }

        static void copy_from_float(Type* p1, float* p2, unsigned int dimensions) {
            for (unsigned int i = 0; i < dimensions; i++) {
                p1[i] = to_16bit_fixed_point(p2[i]);
            }
        }

        static void free(Type&) {}

        static std::vector<float> generate_random(unsigned int dimensions) {
            std::normal_distribution<float> normal_distribution(0.0, 1.0);
            auto& generator = get_default_random_generator();

            std::vector<float> values;
            for (unsigned int i=0; i<dimensions; i++) {
                values.push_back(normal_distribution(generator));
            }
            return values;
        }

        static void serialize_args(std::ostream& out, const Args& args) {
            out.write(reinterpret_cast<const char*>(&args), sizeof(Args));
        }

        static void deserialize_args(std::istream& in, Args* args) {
            in.read(reinterpret_cast<char*>(args), sizeof(Args));
        }

        static void serialize_type(std::ostream& out, const Type& type) {
            out.write(reinterpret_cast<const char*>(&type), sizeof(Type));
        }

        static void deserialize_type(std::istream& in, Type* type) {
            in.read(reinterpret_cast<char*>(type), sizeof(Type));
        }
    };

    template <>
    std::vector<float> convert_stored_type<UnitVectorFormat, std::vector<float>>(
        typename UnitVectorFormat::Type* storage,
        DatasetDescription<UnitVectorFormat> dataset
    ) {
        std::vector<float> res;
        res.reserve(dataset.args);
        for (size_t i=0; i < dataset.args; i++) {
            res.push_back(UnitVectorFormat::from_16bit_fixed_point(storage[i]));
        }
        return res;
    }
}
