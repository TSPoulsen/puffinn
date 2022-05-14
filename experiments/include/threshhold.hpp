#include <iostream>
#include <cstdio>
#include <sstream>
#include <H5Cpp.h>
#include <cassert>

#include "utils.hpp"
#include <puffinn.hpp>
#include "puffinn/pq_filter.hpp"
#include "puffinn/kmeans.hpp"
#include "puffinn/math.hpp"
#include "puffinn/format/generic.hpp"
#include "puffinn/filterer.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash_source/independent.hpp"
#include <puffinn/dataset.hpp>



#define DEFAULT_DATA "data/glove-100-angular.hdf5"
using namespace puffinn;


void write_results(float *estimates, float *true_inner, H5::Group *grp, H5::DataSpace space) {
    H5::DataSet *eip = new H5::DataSet(grp->createDataSet("estimated_inner", H5::PredType::NATIVE_FLOAT, space));
    eip->write(estimates, H5::PredType::NATIVE_FLOAT);

    H5::DataSet *tip = new H5::DataSet(grp->createDataSet("true_inner", H5::PredType::NATIVE_FLOAT, space));
    tip->write(true_inner, H5::PredType::NATIVE_FLOAT);
}




// Calculates the true inner product between queries and dataset
// as well as estimated inner products using PQ for varying values of m [1,2,4,8]
void pq_passing_filter(const unsigned int M = 8, const std::string loss = "mahalanobis", const std::string data_path = DEFAULT_DATA, const bool perm = false)
{
    std::cerr << "START PQ FILTER PASSING" << std::endl << std::endl;
#ifdef __AVX2__
    std::cerr << "USING AVX2" << std::endl;
#endif
    KMeans::distanceType dtype;
    if (loss == "mahalanobis") dtype = KMeans::mahalanobis;
    else if (loss == "euclidean") dtype = KMeans::euclidean;
    else assert(false);
    struct utils::Timer t;
    t.start();
    // setup
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", data_path, 500000);
    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    if (perm)
        train.permute();
    for (auto &v : train_v) {
        train.insert(v);
    }

    std::vector<std::vector<float>> test_v;
    std::pair<int,int> test_dim = utils::load(test_v, "test", data_path, 1000);

    std::stringstream ss;
    int start = data_path.find("/") + 1,
        end   = data_path.find(".");
    ss << "experiments/results/" << data_path.substr(start, end - start ) << "_" << loss << "_" << M;
    if (perm) ss << "_perm.hdf5";
    else ss << "_no_perm.hdf5";
    std::cout << ss.str() << std::endl;
    H5::H5File *file = new H5::H5File(ss.str(), H5F_ACC_TRUNC);

    unsigned int section_size = 50;
    float *estimated_inner = new float[section_size * train_dim.first];
    float *true_inner = new float[section_size * train_dim.first];

    puffinn::PQFilter filter(train, M, 256u, dtype);
    filter.rebuild();
    std::cout << "rebuild done" << std::endl;

    hsize_t dims[2] = {section_size, train_dim.first};
    H5::DataSpace space(2, dims);
    std::cout << "bootthresh done" << std::endl;
    // Should be interpreted as a (test_dim.first , train_dim.first) array
    // The array that all results are written to, so the same size array doesn't have to be reallocated all the time
    std::cout << "query: ";
    unsigned int section = 0;
    for (int j = 0; j < test_dim.first; j++) {
        if (j%section_size == 0){
            std::cout << j << "-" << std::flush; 
            std::ostringstream ss_sec; ss_sec << "section" << section;
            H5::Group *grp = new H5::Group(file->createGroup(ss_sec.str()));
            write_results(estimated_inner, true_inner, grp, space);
            section++;

        } 
        int16_t *q = to_stored_type<UnitVectorFormat>(test_v[j], train.get_description()).get();
        filter.precomp_query_to_centroids(q);
        for (unsigned int i = 0; i < train_dim.first; i++) {
            estimated_inner[(j%section_size)*train_dim.first + i] = filter.estimatedInnerProduct(i);
            true_inner[(j%section_size)*train_dim.first + i] = UnitVectorFormat::from_16bit_fixed_point(dot_product_i16(q, train[i], train.get_description().storage_len));
        }
    }
    std::ostringstream ss_sec; ss_sec << "section" << section;
    H5::Group *grp = new H5::Group(file->createGroup(ss_sec.str()));
    write_results(estimated_inner, true_inner, grp, space);

    delete[] estimated_inner;
    delete[] true_inner;
    float execution_time = t.duration();
    std::cerr << std::endl << "EXPERIMENT TOOK: " << execution_time << "s" << std::endl;
}


void lsh_passing_filter(std::string data_path = DEFAULT_DATA, const int n_sketches = 1)
{
    std::cout << "START LSH FILTER PASSING" << std::endl << std::endl;
    std::cout << data_path << " " << n_sketches << std::endl;

    std::stringstream ss;
    int start = data_path.find("/") + 1,
        end   = data_path.find(".");
    ss << "experiments/results/" << data_path.substr(start, end - start) << "_lsh_" << std::to_string(n_sketches) << ".hdf5" ;
    std::cout << ss.str() << std::endl;
    H5::H5File *file = new H5::H5File(ss.str(), H5F_ACC_TRUNC);
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", data_path, 5000000);
    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    for (auto &v : train_v) {
        train.insert(v);
    }

    std::vector<std::vector<float>> test_v;
    std::pair<int,int> test_dim = utils::load(test_v, "test", data_path, 100);

    int *diffs = new int[test_dim.first * train_dim.first];
    std::fill_n(diffs, test_dim.first * train_dim.first, n_sketches*64);

    for (unsigned int s_i = 0; s_i < n_sketches; s_i++) {
        Filterer<SimHash> filter(IndependentHashArgs<SimHash>(), train.get_description());
        filter.add_sketches(train, 0);
        std::cout << "Calculating diffs" << std::endl;
        for (unsigned int q_i = 0; q_i < test_dim.first; q_i++) {
            std::cout << q_i << "-" << std::flush;
            int16_t *q = to_stored_type<UnitVectorFormat>(test_v[q_i], train.get_description()).get();
            QuerySketches sketches = filter.reset(q);
            for (uint32_t i = 0; i < train_dim.first; i++) {
                //std::cout << "\tsketch_idx: " << sketch_idx << std::endl;
                int_fast32_t sketch_idx = 0; // (q_i + i) % 32;
                FilterLshDatatype sketch = filter.get_sketch(i, sketch_idx);
                FilterLshDatatype q_sketch = sketches.query_sketches[sketch_idx];
                diffs[train_dim.first*q_i + i] -= __builtin_popcountll(sketch ^ q_sketch);
            }
        }
    }
    std::cout << std::endl;
    hsize_t dims[2] = {test_dim.first, train_dim.first};
    H5::DataSpace space(2, dims);
    H5::DataSet *b_diffs = new H5::DataSet(file->createDataSet("collisions", H5::PredType::NATIVE_INT, space));
    b_diffs->write(diffs, H5::PredType::NATIVE_INT);

    delete[] diffs;
    float *tip = new float[test_dim.first * train_dim.first];
    for (int j = 0; j < test_dim.first; j++) {
        std::cout << "query: " << j << std::endl;
        auto q = to_stored_type<UnitVectorFormat>(test_v[j], train.get_description()).get();
        for (int i = 0; i < train_dim.first; i++) {
            tip[j*train_dim.first + i] = UnitVectorFormat::from_16bit_fixed_point(dot_product_i16(q, train[i], train.get_description().storage_len));
        }
    }

    H5::DataSet *real_inner = new H5::DataSet(file->createDataSet("true_inner", H5::PredType::NATIVE_FLOAT, space));
    real_inner->write(tip, H5::PredType::NATIVE_FLOAT);

    delete[] tip;
}

void run_pass_filter(int argc, char *argv[]) 
{
    std::cout << std::endl << std::endl;
    if (argc == 5) {
        const std::string data_path = argv[1];
        const unsigned int M = (unsigned int) atoi(argv[2]);
        const std::string loss = argv[3];
        const std::string perm_arg = argv[4];
        const bool perm = perm_arg == "perm";

        pq_passing_filter(M, loss, data_path, perm);
    }
    else if (argc == 3) {
        const std::string data_path = argv[1];
        const int n_sketches = atoi(argv[2]);
        lsh_passing_filter(data_path, n_sketches);
    }

}