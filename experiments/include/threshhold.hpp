#include <iostream>
#include <sstream>
#include <H5Cpp.h>

#include "utils.hpp"
#include <puffinn.hpp>
#include "puffinn/pq_filter.hpp"
#include "puffinn/kmeans.hpp"
#include "puffinn/math.hpp"
#include "puffinn/format/generic.hpp"
#include "puffinn/filterer.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash_source/independent.hpp"



#define DATA_PATH "data/glove-100-angular.hdf5"
using namespace puffinn;

// Calculates the true inner product between queries and dataset
// as well as estimated inner products using PQ for varying values of m [1,2,4,8]
void pq_passing_filter()
{
    std::cerr << "START PQ FILTER PASSING" << std::endl << std::endl;
#ifdef __AVX2__
    std::cerr << "USING AVX2" << std::endl;
#endif
    struct utils::Timer t;
    t.start();
    // setup
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", DATA_PATH);
    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    train.dont_permute();
    for (auto &v : train_v) {
        train.insert(v);
    }

    std::vector<std::vector<float>> test_v;
    std::pair<int,int> test_dim = utils::load(test_v, "test", DATA_PATH, 100);

    H5::H5File *file = new H5::H5File("experiments/results/g100_pass_filter_maha16_no_perm2.hdf5", H5F_ACC_TRUNC);

    float *estimate_arr = new float[test_dim.first * train_dim.first];
    float *real_arr = new float[test_dim.first * train_dim.first];

    puffinn::PQFilter filter(train, 16, 256, KMeans::mahalanobis);
    filter.rebuild();

    // First different bootstrapped thresholds
    size_t bt_size = 6;
    int boot_k[bt_size] = {1, 10, 15, 20, 50, 100};
    float *boot_thresh = new float[bt_size];

    for (size_t i = 0; i < bt_size; i++) {
        boot_thresh[i]= filter.bootStrapThreshold(100u, 5000u, boot_k[i]);
    }
    hsize_t dimst[1] = {bt_size};
    H5::DataSpace spacet(1, dimst);
    H5::DataSet *boot_k_d = new H5::DataSet(file->createDataSet("threshholds_boot_k", H5::PredType::NATIVE_INT, spacet));
    boot_k_d->write(boot_k, H5::PredType::NATIVE_INT);
    H5::DataSet *threshs = new H5::DataSet(file->createDataSet("threshholds", H5::PredType::NATIVE_FLOAT, spacet));
    threshs->write(boot_thresh, H5::PredType::NATIVE_FLOAT);

    hsize_t dims[2] = {test_dim.first, train_dim.first};
    H5::DataSpace space(2, dims);
    // Should be interpreted as a (test_dim.first , train_dim.first) array
    // The array that all results are written to, so the same size array doesn't have to be reallocated all the time
    for (int j = 0; j < test_dim.first; j++) {
        auto q = to_stored_type<UnitVectorFormat>(test_v[j], train.get_description()).get();
        filter.precomp_query_to_centroids(q);
        for (int i = 0; i < train_dim.first; i++) {
            estimate_arr[j*train_dim.first + i] = filter.estimatedInnerProduct(i);
            real_arr[j*train_dim.first + i] = UnitVectorFormat::from_16bit_fixed_point(dot_product_i16(q, train[i], train.get_description().storage_len));
        }
    }

    H5::DataSet *estimated_inner = new H5::DataSet(file->createDataSet("estimated_inner", H5::PredType::NATIVE_FLOAT, space));
    estimated_inner->write(estimate_arr, H5::PredType::NATIVE_FLOAT);

    H5::DataSet *real_inner = new H5::DataSet(file->createDataSet("true_inner", H5::PredType::NATIVE_FLOAT, space));
    real_inner->write(real_arr, H5::PredType::NATIVE_FLOAT);


    delete[] estimate_arr;
    delete[] real_arr;
    float execution_time = t.duration();
    std::cerr << std::endl << "EXPERIMENT TOOK: " << execution_time << "s" << std::endl;
}


void lsh_passing_filter()
{
    std::cout << "START LSH FILTER PASSING" << std::endl << std::endl;

    H5::H5File *file = new H5::H5File("experiments/results/g100_pass_filter_lshq1.hdf5", H5F_ACC_TRUNC);
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", DATA_PATH);
    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    train.dont_permute(); // Shouldn't make a difference here
    for (auto &v : train_v) {
        train.insert(v);
    }

    std::vector<std::vector<float>> test_v;
    std::pair<int,int> test_dim = utils::load(test_v, "test", DATA_PATH, 100);

    int *diffs = new int[NUM_SKETCHES * train_dim.first];

    Filterer<SimHash> filter(IndependentHashArgs<SimHash>(), train.get_description());
    filter.add_sketches(train, 0);
    std::cout << "Calculating diffs" << std::endl;
    for (unsigned int q_i = 0; q_i < test_dim.first; q_i++) {
        std::cout << q_i << "-";
        int16_t *q = to_stored_type<UnitVectorFormat>(test_v[q_i], train.get_description()).get();
        QuerySketches sketches = filter.reset(q);
        for (unsigned int i = 0; i < train_dim.first; i++) {
            for (unsigned int sketch_idx = 0; sketch_idx < NUM_SKETCHES; sketch_idx++) {
                uint64_t sketch = filter.get_sketch(i, sketch_idx);
                uint64_t q_sketch = sketches.query_sketches[sketch_idx];
                diffs[NUM_SKETCHES*i + sketch_idx]  = popcountll(sketch ^ q_sketch);
            }
        }
        break;
    }
    std::cout << std::endl;
    hsize_t dims[2] = {NUM_SKETCHES, train_dim.first};
    H5::DataSpace space(2, dims);
    H5::DataSet *b_diffs = new H5::DataSet(file->createDataSet("bit_diffs", H5::PredType::NATIVE_INT, space));
    b_diffs->write(diffs, H5::PredType::NATIVE_INT);

    delete[] diffs;
}