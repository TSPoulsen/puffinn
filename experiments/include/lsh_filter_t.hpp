#include <puffinn.hpp>
#include "puffinn/filterer.hpp"
#include "puffinn/hash/simhash.hpp"
#include "puffinn/hash_source/independent.hpp"
#include "utils.hpp"

#include <H5Cpp.h>
#include <iostream>

#define DATA_PATH "data/glove-100-angular.hdf5"
using namespace puffinn;


int main() {
    std::vector<std::vector<float>> train_v;
    std::pair<int,int> train_dim = utils::load(train_v, "train", DATA_PATH);
    Dataset<UnitVectorFormat> train(train_dim.second,train_dim.first);
    //train.permute(); // Shouldn't make a difference here
    for (auto &v : train_v) {
        train.insert(v);
    }
    Filterer<SimHash> filter(IndependentHashArgs<SimHash>(), train.get_description());
    filterer.add_sketches(dataset, last_rebuild);


}





