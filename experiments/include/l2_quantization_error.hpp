
#include <iostream>
#include <puffinn.hpp>
#include <H5Cpp.h>
#include "utils.hpp"
#include "puffinn/pq_filter.hpp"

// Caclulates the quantization error in l2 distance between quantized representation of vector
// agains the the actual vector
void quant_error()
{
    std::cerr << "START QUANTIZATION_ERROR EXPERIMENT" << std::endl << std::endl;
    struct utils::Timer t;
    t.start();
    // Code to be timed here
    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-100-angular.hdf5";
    auto dims = utils::load(data, "train", data_path);
    puffinn::Dataset<puffinn::UnitVectorFormat> ds(data[0].size(), data.size());    
    for(std::vector<float> v: data){
        ds.insert(v);
    }
    int runs = 3, m = 8, k = 256;
    puffinn::PQFilter pq1(ds, m, k, puffinn::KMeans::mahalanobis);
    std::cerr << "Calculating results" << std::endl;
    for (int i = 0; i < runs; i++) {
        pq1.rebuild();
        std::cout<< "RUN# " << i << " QUANTIZATION ERROR: " << pq1.getQuantizationError() << std::endl;
    }
        
    // std::cerr << "Creating file for results" << std::endl;
    // hsize_t dims[2] = {data_dimensions.first};
    // // number of dimensions and the size of each dimension
    // H5::DataSpace space(2, dims);
    // H5::H5File *file = new H5::H5File("experiments/results/quant_error.hdf5", H5F_ACC_TRUNC);
    // H5::DataSet *dataset = new H5::DataSet(file->createDataSet("Quantization_error_l2", H5::PredType::NATIVE_FLOAT, space));

    // dataset->write(results, H5::PredType::NATIVE_FLOAT);


    float execution_time = t.duration();
    std::cerr << std::endl << "EXPERIMENT TOOK: " << execution_time << "s" << std::endl;
}