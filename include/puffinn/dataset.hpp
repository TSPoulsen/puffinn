#pragma once

#include "puffinn/format/generic.hpp"
#include "puffinn/typedefs.hpp"
#include <cstring>
#include <assert.h>
#include <vector>
#include <istream>
#include <memory>
#include <algorithm>
#include <ostream>

namespace puffinn {

    // Fisher–Yates_shuffle
    // If sample_size == data_size, it returns a random permutation of indicies
    std::vector<unsigned int> random_sample(unsigned int sample_size, unsigned int data_size)
    {
        auto &gen = get_default_random_generator();
        assert(sample_size <= data_size);
        std::vector<unsigned int> res(sample_size);
        for (unsigned int i = 0; i != data_size; ++i) {
            std::uniform_int_distribution<unsigned int> dis(0, i);
            std::size_t j = dis(gen);
            if (j < res.size()) {
                if (i < res.size()) {
                    res[i] = res[j];
                }
                res[j] = i;
            }
        }
        for (unsigned int idx : res) {
            assert(idx < data_size);
        }
        return res;
    }

    const unsigned int DEFAULT_CAPACITY = 100;
    const float EXPANSION_FACTOR = 1.5;

    // The container for all inserted vectors.
    // The data is stored according to the given format.
    template <typename T>
    class Dataset {
        // Arguments 
        typename T::Args args;
        // Number of dimensions of stored vectors, which may be padded to
        // more easily map to simd instructions.
        unsigned int storage_len;
        // Number of inserted vectors.
        unsigned int inserted_vectors;
        // Maximal number of inserted vectors.
        unsigned int capacity;
        // Inserted vectors, aligned to the vector alignment.
        AlignedStorage<T> data;
        // Permutation of vectors before alignment
        std::vector<unsigned int> permutation;

    public:
        // Create an empty storage for vectors with the given number of dimensions.
        Dataset(typename T::Args args) : Dataset(args, DEFAULT_CAPACITY)
        {
        }

        // Create an empty storage for vectors with the given number of dimensions.
        // Allocates enough space for the given number of vectors before needing to reallocate.
        Dataset(typename T::Args args, unsigned int capacity)
          : args(args),
            storage_len(pad_dimensions<T>(T::storage_dimensions(args))),
            inserted_vectors(0),
            capacity(capacity),
            data(allocate_storage<T>(capacity, storage_len)),
            permutation(random_sample(args, args))
        {
        }

        Dataset(Dataset&& other)
          : args(other.args),
            storage_len(other.storage_len),
            inserted_vectors(other.inserted_vectors),
            capacity(other.capacity),
            permutation(other.permutation),
            data(std::move(other.data))
        {
        }

        Dataset& operator=(Dataset& rhs) {
            if (this != &rhs) {
                args = rhs.args;
                storage_len = rhs.storage_len;
                inserted_vectors = rhs.inserted_vectors;
                capacity = rhs.capacity;
                permutation = rhs.permutation;
                data = std::move(rhs.data);
            }
            return *this;
        }

        Dataset& operator=(Dataset&& rhs) {
            if (this != &rhs) {
                args = rhs.args;
                storage_len = rhs.storage_len;
                inserted_vectors = rhs.inserted_vectors;
                capacity = rhs.capacity;
                permutation = rhs.permutation;
                data = std::move(rhs.data);
            }
            return *this;
        }

        Dataset(std::istream& in) {
            T::deserialize_args(in, &args);
            in.read(reinterpret_cast<char*>(&storage_len), sizeof(unsigned int));
            in.read(reinterpret_cast<char*>(&inserted_vectors), sizeof(unsigned int));
            capacity = inserted_vectors;
            data = allocate_storage<T>(capacity, storage_len);
            for (size_t i=0; i < inserted_vectors*storage_len; i++) {
                T::deserialize_type(in, &data.get()[i]);
            }
        }

        void serialize(std::ostream& out) const {
            T::serialize_args(out, args);
            out.write(reinterpret_cast<const char*>(&storage_len), sizeof(unsigned int));
            out.write(reinterpret_cast<const char*>(&inserted_vectors), sizeof(unsigned int));
            for (size_t i=0; i < inserted_vectors*storage_len; i++) {
                T::serialize_type(out, data.get()[i]);
            }
        }
        // creates the normal ordering for dimensions
        void dont_permute() {
            std::sort(permutation.begin(), permutation.end()); 
        }

        // Access the vector at the given position.
        typename T::Type* operator[](unsigned int idx) const {
            return &data.get()[idx*storage_len];
        }

        // Retrieve the number of dimensions of vectors inserted into this dataset,
        // as well as the number of dimensions they are stored with.
        DatasetDescription<T> get_description() const {
            DatasetDescription<T> res;
            res.args = args;
            res.storage_len = storage_len;
            res.permutation = permutation;
            return res;
        }

        // Retrieve the number of inserted vectors.
        unsigned int get_size() const {
            return inserted_vectors;
        }

        // Insert a vector.
        template <typename U>
        void insert(const U& vec) {
            if (inserted_vectors == capacity) {
                unsigned int new_capacity = std::ceil(capacity*EXPANSION_FACTOR);
                auto new_data = allocate_storage<T>(new_capacity, storage_len);
                for (size_t i=0; i < capacity*storage_len; i++) {
                    new_data.get()[i] = std::move(data.get()[i]);
                }
                data = std::move(new_data);
                capacity = new_capacity;
            }
            T::store(
                vec,
                &data.get()[inserted_vectors*storage_len],
                get_description());
            inserted_vectors++;
        }

        // Retrieve the capacity of the dataset
        unsigned int get_capacity() const {
            return capacity;
        }

        // Remove all points from the dataset.
        void clear() {
            inserted_vectors = 0;
        }

        uint64_t memory_usage() const {
            uint64_t inner_memory = 0;
            for (size_t i=0; i < inserted_vectors*storage_len; i++) {
                inner_memory += T::inner_memory_usage(data.get()[i]); 
            }
            return sizeof(Dataset<T>)
                + capacity*storage_len*sizeof(typename T::Type)
                + inner_memory;
        }
    };
}
