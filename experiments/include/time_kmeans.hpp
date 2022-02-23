
#include <H5Cpp.h>

#include "utils.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/kmeans.hpp"

#define DATA_PATH "data/glove-25-angular.hdf5"

using namespace puffinn;
// Runs and times a single kmeans 'run' mutiple times
void time_kmeans()
{
    std::cout << "START time_kmeans.hpp" << std::endl << std::endl;
    struct utils::Timer t;
    // setup
    Dataset<UnitVectorFormat> train(0,0);
    std::pair<int,int> train_dim = utils::load<UnitVectorFormat>(train, "train", DATA_PATH, 10000);
    puffinn::PQFilter filter(train, 1);

    H5::H5File *file = new H5::H5File("experiments/results/time_kmeans_V1.hdf5", H5F_ACC_TRUNC);

    int runs = 25;
    hsize_t dims[1] = {runs};
    H5::DataSpace space(1, dims);

    float times[25];
    std::vector<std::vector<float>> data = filter.getSubspace(0);
    KMeans clusterer(256);
    for (int r = 0; r < runs; r++) {
        std::cout << "RUN " << r << std::endl;
        t.start();
        clusterer.fit(data);
        times[r] = t.duration();
    }
    H5::DataSet *times_data = new H5::DataSet(file->createDataSet("times_data", H5::PredType::NATIVE_FLOAT, space));
    times_data->write(times, H5::PredType::NATIVE_FLOAT);
}
