#include <iostream>
#include <sstream>
#include <puffinn.hpp>
#include <H5Cpp.h>

#include "utils.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/math.hpp"

#define DATA_PATH "data/glove-25-angular.hdf5"
using namespace puffinn;

// Calculates the true inner product between queries and dataset
// as well as estimated inner products using PQ for varying values of m [1,2,4,8]
void ang_dist_glove_subset()
{
    std::cerr << "Euclidean distance vs Mahalanobis distance experiment" << std::endl << std::endl;
    // setup
    Dataset<UnitVectorFormat> train(0,0);
    Dataset<UnitVectorFormat> test(0,0);
    std::pair<int,int> train_dim = utils::load<UnitVectorFormat>(train, "train", DATA_PATH, 100);
    std::pair<int,int> test_dim = utils::load<UnitVectorFormat>(test, "test", DATA_PATH, 10);
    H5::H5File *file = new H5::H5File("experiments/results/glove-25-100k-100.hdf5", H5F_ACC_TRUNC);

    hsize_t dims[2] = {test_dim.first, train_dim.first};
    H5::DataSpace space(2, dims);

    // First calculate true answer

    // Should be interpreted as a (test_dim.first , train_dim.first) array
    // The array that all results are written to, so the same size array doesn't have to be reallocated all the time
    float *result_arr = new float[test_dim.first * train_dim.first];
    for (int j = 0; j < test_dim.first; j++) {
        for (int i = 0; i < train_dim.first; i++) {
            result_arr[j*train_dim.first + i] = UnitVectorFormat::from_16bit_fixed_point(dot_product_i16(test[j], train[i], train.get_description().storage_len));
        }
    }
    H5::DataSet *true_inner_data = new H5::DataSet(file->createDataSet("True_innner_product", H5::PredType::NATIVE_FLOAT, space));
    true_inner_data->write(result_arr, H5::PredType::NATIVE_FLOAT);

    // Calculating & writing results
    for (int m = 1; m < 9; m*=2) {
        std::cerr << "RUN M=" << m << std::endl;
        std::ostringstream ss; ss << m << "m";
        H5::Group *grp = new H5::Group(file->createGroup(ss.str()));
        puffinn::PQFilter filter_euc(train, puffinn::KMeans::euclidean, m, 256);
        puffinn::PQFilter filter_maha(train, puffinn::KMeans::euclidean, m, 256);

        // calculate asymmetric distance for euclidean PQFilter
        for (int j = 0; j < test_dim.first; j++) {
            for (int i = 0; i < train_dim.first; i++) {
                result_arr[j*train_dim.first + i] = filter_euc.asymmetricDistanceComputation(test[j], train[i]);
            }
        }
        H5::DataSet *asym_data = new H5::DataSet(grp->createDataSet("Asymmetric_distance_euclidean", H5::PredType::NATIVE_FLOAT, space));
        asym_data->write(result_arr, H5::PredType::NATIVE_FLOAT);

        // calculate asymmetric distance for mahalanobis PQFilter
        for (int j = 0; j < test_dim.first; j++) {
            for (int i = 0; i < train_dim.first; i++) {
                result_arr[j*train_dim.first + i] = filter_maha.asymmetricDistanceComputation(test[j], train[i]);
            }
        }
        H5::DataSet *sym_data = new H5::DataSet(grp->createDataSet("Symmetric_distance_mahalanobis", H5::PredType::NATIVE_FLOAT, space));
        sym_data->write(result_arr, H5::PredType::NATIVE_FLOAT);
    }

    delete[] result_arr;
}