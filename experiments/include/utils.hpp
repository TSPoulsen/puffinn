#pragma once

#include <vector>
#include <iostream>
#include <valarray>
#include <cassert>
#include <climits>
#include <H5Cpp.h>
#include <string>
#include <chrono>
#include "puffinn.hpp"


// Relative path to use this dataset
#define DEFAULT_HDF5_FILE "data/glove-25-angular.hdf5"

namespace utils {
    struct Timer {

        using clock_type = std::chrono::steady_clock;
        using second_type = std::chrono::duration<double, std::ratio<1> >;

        std::chrono::time_point<clock_type> _start;

        void start() {
            _start = clock_type::now();
        }
        // Returns time since start() call in seconds
        float duration() {
            return std::chrono::duration_cast<second_type>(clock_type::now() - _start).count();
        }

    };

    // This code is 'inspired' by https://github.com/Cecca/running-experiments/blob/master/datasets.hpp
    template<typename TFormat>
    std::pair<int, int> load(puffinn::Dataset<TFormat>& dataset, std::string set, std::string path = DEFAULT_HDF5_FILE, int max_size = INT_MAX) 
    {

        // Open the file and get the dataset
        //std::cerr << "LOADING FILE" << std::endl;
        H5::H5File file(path, H5F_ACC_RDONLY);
        H5::Group group = file.openGroup("/");
        H5::DataSet h5_dataset = group.openDataSet(set);

        H5T_class_t type_class = h5_dataset.getTypeClass();
        // Check that we have a dataset of floats
        if (type_class != H5T_FLOAT) { throw std::runtime_error("wrong type class"); }
        H5::DataSpace dataspace = h5_dataset.getSpace();

        // Get the number of vectors and the dimensionality
        hsize_t data_dims[2];
        dataspace.getSimpleExtentDims(data_dims, NULL);
        int n = data_dims[0], dim = data_dims[1];
        dataset = puffinn::Dataset<puffinn::UnitVectorFormat>(dim, n);
        dataset.dont_permute();
        std::valarray<float> temp(n * dim);
        h5_dataset.read(&temp[0], H5::PredType::NATIVE_FLOAT);
        if (n > max_size) n = max_size;
        for (int i = 0; i < n; i++ ) {
            std::vector<float> vec(&temp[i*dim], &temp[(i+1)*dim]);
            dataset.insert(vec);
        }
        std::cout << "Loaded "<< set << " dataset of size (" << n << "," << dim << ")" << std::endl;
        return std::make_pair(n, dim);;
    }

    // Load dataset into vector of vectors but have it normalized by loading it into a Dataset<UnitVectorFormat> first
    std::pair<int, int> load(std::vector<std::vector<float>> &data, std::string set, std::string path = DEFAULT_HDF5_FILE, int max_size = INT_MAX)
    {
        assert(data.size() == 0u);
        // Open the file and get the dataset
        //std::cerr << "LOADING FILE" << std::endl;
        H5::H5File file(path, H5F_ACC_RDONLY);
        H5::Group group = file.openGroup("/");
        H5::DataSet h5_dataset = group.openDataSet(set);

        H5T_class_t type_class = h5_dataset.getTypeClass();
        // Check that we have a dataset of floats
        if (type_class != H5T_FLOAT) { throw std::runtime_error("wrong type class"); }
        H5::DataSpace dataspace = h5_dataset.getSpace();

        // Get the number of vectors and the dimensionality
        hsize_t data_dims[2];
        dataspace.getSimpleExtentDims(data_dims, NULL);
        int n = data_dims[0], dim = data_dims[1];
        std::valarray<float> temp(n * dim);
        h5_dataset.read(&temp[0], H5::PredType::NATIVE_FLOAT);
        if (n > max_size) n = max_size;
        for (int i = 0; i < n; i++ ) {
            std::vector<float> vec(&temp[i*dim], &temp[(i+1)*dim]);
            data.push_back(vec);
        }
        std::cout << "Loaded "<< set << " dataset of size (" << n << "," << dim << ")" << std::endl;
        return std::make_pair(n, dim);;
    }

}