#include <H5Cpp.h>
#include <sstream>

#include "utils.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/kmeans.hpp"

#define DATA_PATH "data/glove-100-angular.hdf5"

using namespace puffinn;

void inertia_run() 
{
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", DATA_PATH);
    H5::H5File *file = new H5::H5File("experiments/results/inertia_runs.hdf5", H5F_ACC_TRUNC);

    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    train.dont_permute(); // Uncomment this to remove random permutations of dimensions
    for (auto &v : train_v) {
        train.insert(v);
    }

    puffinn::PQFilter pq(train, 8, 256);
    KMeans km_e(256, KMeans::euclidean);
    KMeans km_m(256, KMeans::mahalanobis);

    int runs = 200;
    double inertias_e[8*runs] = {0};
    double inertias_m[8*runs] = {0};
    for (int m = 0; m < 8; m++) {
        std::cout << m << std::endl;
        auto data = pq.getSubspace(m);
        km_e.padData(data);
        km_m.createCovarianceMatrix(data);
        auto clusters_e = km_e.init_centroids_kpp(data);
        auto clusters_m = km_e.init_centroids_kpp(data);
        for (int r = 0; r < runs; r++) {
            inertias_e[200*m + r] += km_e.lloyd(data, clusters_e, 1);
            inertias_m[200*m + r] += km_m.lloyd(data, clusters_m, 1);
        }
    }
    std::cout << "Done" << std::endl;
    hsize_t dims[2] = {runs, 8};
    H5::DataSpace space(2, dims);
    H5::DataSet *inertia_de = new H5::DataSet(file->createDataSet("euclidean", H5::PredType::NATIVE_DOUBLE, space));
    H5::DataSet *inertia_dm = new H5::DataSet(file->createDataSet("mahalanobis", H5::PredType::NATIVE_DOUBLE, space));
    inertia_de->write(inertias_e, H5::PredType::NATIVE_DOUBLE);
    inertia_dm->write(inertias_m, H5::PredType::NATIVE_DOUBLE);




}
// Runs and times a single kmeans 'run' mutiple times
void inertia_variance()
{
    std::cout << "START inertia_variance.hpp" << std::endl << std::endl;
    // setup
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", DATA_PATH);

    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    train.dont_permute(); // Uncomment this to remove random permutations of dimensions
    for (auto &v : train_v) {
        train.insert(v);
    }

    puffinn::PQFilter filter(train, 8, 256, KMeans::mahalanobis);

    H5::H5File *file = new H5::H5File("experiments/results/inertia_variance_maha.hdf5", H5F_ACC_TRUNC);

    int runs = 25;
    hsize_t dims[1] = {runs};
    H5::DataSpace space(1, dims);

    double inertias[25];
    for (int i = 0; i < 8; i++) {
        KMeans clusterer(256, KMeans::mahalanobis, 1);
        for (int r = 0; r < runs; r++) {
            std::vector<std::vector<float>> data = filter.getSubspace(i);
            std::cout << "RUN " << r << std::endl;
            clusterer.fit(data);
            inertias[r] = clusterer.gb_inertia;
        }
        std::cout << std::endl;
        std::ostringstream ss; ss << i << "m";
        H5::DataSet *inertia_data = new H5::DataSet(file->createDataSet(ss.str(), H5::PredType::NATIVE_DOUBLE, space));
        inertia_data->write(inertias, H5::PredType::NATIVE_DOUBLE);
    }
}
