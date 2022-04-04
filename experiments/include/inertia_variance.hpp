#include <H5Cpp.h>
#include <sstream>

#include "utils.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/kmeans.hpp"

#define DATA_PATH "data/glove-25-angular.hdf5"

using namespace puffinn;
// Runs and times a single kmeans 'run' mutiple times
void inertia_variance()
{
    std::cout << "START inertia_variance.hpp" << std::endl << std::endl;
    // setup
    Dataset<UnitVectorFormat> train(0,0);
    std::pair<int,int> train_dim = utils::load<UnitVectorFormat>(train, "train", DATA_PATH, 100000);
    puffinn::PQFilter filter(train, 4);

    H5::H5File *file = new H5::H5File("experiments/results/inertia_variance_maha.hdf5", H5F_ACC_TRUNC);

    int runs = 25;
    hsize_t dims[1] = {runs};
    H5::DataSpace space(1, dims);

    double inertias[25];
    for (int i = 0; i < 4; i++) {
        std::vector<std::vector<float>> data = filter.getSubspace(i);
        KMeans clusterer(256, KMeans::mahalanobis, 1);
        for (int r = 0; r < runs; r++) {
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
