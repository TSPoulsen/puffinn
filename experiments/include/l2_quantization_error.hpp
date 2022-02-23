
#include <iostream>
#include <puffinn.hpp>
#include <H5Cpp.h>
#include "utils.hpp"
#include "puffinn/pq_filter.hpp"

// Caclulates the quantization error in l2 distance between quantized representation of vector
// agains the the actual vector
void l2_quant_error()
{
    std::cerr << "START L2_QUANTIZATION_ERROR EXPERIMENT" << std::endl << std::endl;
    struct utils::Timer t;
    puffinn::Dataset<puffinn::UnitVectorFormat> train(0,0);
    std::pair<int,int> data_dimensions = utils::load(train, "train");
    t.start();
    // Code to be timed here
    puffinn::PQFilter filter(train, 5, 256);

    std::cerr << "Calculating results" << std::endl;
    float results[data_dimensions.first];
    for (int i = 0; i < data_dimensions.first; i++) {
        results[i] = filter.quantizationError(i);
    }
    
    
    std::cerr << "Creating file for results" << std::endl;
    hsize_t dims[1] = {data_dimensions.first};
    // number of dimensions and the size of each dimension
    H5::DataSpace space(1, dims);
    H5::H5File *file = new H5::H5File("experiments/results/l2_quant_error.hdf5", H5F_ACC_TRUNC);
    H5::DataSet *dataset = new H5::DataSet(file->createDataSet("Quantization_error_l2", H5::PredType::NATIVE_FLOAT, space));

    dataset->write(results, H5::PredType::NATIVE_FLOAT);


    float execution_time = t.duration();
    std::cerr << std::endl << "EXPERIMENT TOOK: " << execution_time << "s" << std::endl;
}