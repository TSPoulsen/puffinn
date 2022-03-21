#include <nanobench.h>
#include <puffinn/kmeans.hpp>
#include <puffinn/pq_filter.hpp>
#include <puffinn/collection.hpp>
#include <puffinn.hpp>
#include <math.h>
#include "utils.hpp"
#include <string>


void mahaBench(ankerl::nanobench::Bench *bencher)
{
    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    auto dims = utils::load(data, "train", data_path, 10);
    puffinn::KMeans km(1, puffinn::KMeans::mahalanobis);
    km.padData(data);
    km.createCovarianceMatrix(data);

    bencher->run("Maha distance WITH AVX", [&] {
        ankerl::nanobench::doNotOptimizeAway(km.mahaDistance_avx(data[0], data[1]));
    });    

    bencher->run("Maha distance NO AVX", [&] {
        ankerl::nanobench::doNotOptimizeAway(km.mahaDistance(data[0], data[1]));
    });    

}

void eucBench(ankerl::nanobench::Bench *bencher)
{
    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    auto dims = utils::load(data, "train", data_path, 10);
    puffinn::KMeans km(1, puffinn::KMeans::mahalanobis);
    km.padData(data);

    bencher->run("Euc distance WITH AVX", [&] {
        ankerl::nanobench::doNotOptimizeAway(km.sumOfSquares_avx(data[0], data[1]));
    });    

    bencher->run("Euc distance NO AVX", [&] {
        ankerl::nanobench::doNotOptimizeAway(km.sumOfSquares(data[0], data[1]));
    });    

}


void kmeansBench(ankerl::nanobench::Bench *bencher)
{
    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    auto dims = utils::load(data, "train", data_path, 1000);
    puffinn::KMeans km_maha(256, puffinn::KMeans::mahalanobis, 1, 100);
    puffinn::KMeans km_euc(256, puffinn::KMeans::euclidean, 1, 100);
    km_maha.padData(data);
    bencher->run("Kmeans mahalanobis 1 run, 100 max iter, 1k points, k=256", [&] {
        km_maha.fit(data);
    });    
    bencher->run("Kmeans euclidean 1 run, 100 max iter, 1k points, k=256", [&] {
        km_euc.fit(data);
    });    

}

void pqfBench(ankerl::nanobench::Bench *bencher)
{
    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    auto dims = utils::load(data, "train", data_path, 10000);
    puffinn::Dataset<puffinn::UnitVectorFormat> ds(data[0].size(), data.size());    
    for(std::vector<float> v: data){
        ds.insert(v);
    }
    puffinn::PQFilter pq1(ds, 2, 256);
    pq1.rebuild();
    alignas(32) int16_t tmp[pq1.getPadSize()];
    pq1.createPaddedQueryPoint(ds[110], tmp);

    bencher->run("Asymmetric computing PQ code every call", [&] {
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation_simple(ds[0], ds[110]));
    });    
    
    bencher->run("Asymmetric PQ code precomputed", [&] {
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation(0u, ds[110]));
    });    
    
    bencher->run("Asymmetric fast creating padded query once", [&] {
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation_avx(0u, tmp));
    });

    bencher->run("Asymmetric fast creating padded query before each call", [&] {
        alignas(32) int16_t tmp1[pq1.getPadSize()];
        pq1.createPaddedQueryPoint(ds[110], tmp1);
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation_avx(0u, tmp1));
    });

    
    bencher->run("building query distances", [&] {
        pq1.precomp_query_to_centroids(tmp);
    });

    bencher->run("Estimated Inner product O(M)", [&] {
        for(unsigned int i = 0; i < 10000; i++){
            ankerl::nanobench::doNotOptimizeAway(pq1.estimatedInnerProduct(0u));
        }
    });


    bencher->run("True Inner product", [&] {
        for(unsigned int i = 0; i < 10000; i++){
            ankerl::nanobench::doNotOptimizeAway(puffinn::dot_product_i16_avx2(ds[0], ds[110], ds.get_description().storage_len));
        }
    });

}
void imp(ankerl::nanobench::Bench *bencher){

    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    auto dims = utils::load(data, "train", data_path, 200000);
    puffinn::Index<puffinn::CosineSimilarity> index(dims.second, 600*1024*1024);
    for (std::vector<float> & v : data) { index.insert(v); }
    index.rebuild();
    std::vector<float> query = {-0.633   , -0.33511 ,  0.52545 ,  0.092909, -0.97386 , -1.3496  ,
       -1.9191  , -0.76974 ,  0.34711 , -1.1012  , -0.31359 ,  0.66227 ,
        0.11019 , -0.70333 ,  1.0159  , -0.1288  ,  0.37742 ,  0.35706 ,
       -0.1153  ,  0.19528 ,  0.36092 ,  0.92362 , -0.92318 ,  0.42094 ,
        0.4587};
      
    bencher->run("search with Simple", [&] {
        std::vector<uint32_t> result = index.search(query, 10, 0.9, puffinn::FilterType::Simple);
    });
    bencher->run("search with None", [&] {
        std::vector<uint32_t> result = index.search(query, 10, 0.9, puffinn::FilterType::None);
    });
    bencher->run("search with PQ_simple", [&] {
        std::vector<uint32_t> result = index.search(query, 10, 0.9, puffinn::FilterType::PQ_Simple);
    });

    /*   
    std::vector<uint32_t> result = index.search(query, 10, 0.95, puffinn::FilterType::PQ_Simple);
    for(auto &ele : std::set<uint32_t>(result.begin(), result.end())){
        std::cout << ele << std::endl;
    }
    */   
}

void all_bench()
{
    ankerl::nanobench::Bench bencher;
    bencher.minEpochIterations(2000);
    imp(&bencher);
    //mahaBench(&bencher);
    //eucBench(&bencher);

    //bencher.minEpochIterations(10);
    //bencher.minEpochIterations(10);
    //pqfBench(&bencher);
    //kmeansBench(&bencher);
}