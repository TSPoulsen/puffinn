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
    auto dims = utils::load(data, "train", data_path, 50000);
    puffinn::Dataset<puffinn::UnitVectorFormat> ds(data[0].size(), data.size());    
    for(std::vector<float> v: data){
        ds.insert(v);
    }
    puffinn::PQFilter pq1(ds, 4, 256);
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
    auto dims = utils::load(data, "train", data_path, 500000);
    puffinn::Index<puffinn::CosineSimilarity> index(dims.second, 1024*1024*1024, true);
    for (std::vector<float> & v : data) { index.insert(v); }
    index.rebuild();  
    bencher->run("search with Simple", [&] {
        std::vector<uint32_t> result = index.search(data[1420], 20, 0.1, puffinn::FilterType::Simple);
    });
    //bencher->run("search with None", [&] {
        //std::vector<uint32_t> result = index.search(query, 10, 0.1, puffinn::FilterType::None);
    //});
    bencher->run("search with PQ_simple", [&] {
        std::vector<uint32_t> result = index.search(data[1420], 20, 0.1, puffinn::FilterType::PQ_Simple);
    });

    /*   
    std::vector<uint32_t> result = index.search(query, 10, 0.95, puffinn::FilterType::PQ_Simple);
    for(auto &ele : std::set<uint32_t>(result.begin(), result.end())){
        std::cout << ele << std::endl;
    }
    */   
}

void correctnessMeassurementPQ(){

    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    int n = 500000;
    int topK = 10;
    auto dims = utils::load(data, "train", data_path, n);
    puffinn::Index<puffinn::CosineSimilarity> index(dims.second, 600*1024*1024, true);
    for (std::vector<float> & v : data) { index.insert(v); }
    index.rebuild();  
    for(int i = 0; i <n; i++){
        std::vector<uint32_t> guess = index.search(data[i], topK, 0.9, puffinn::FilterType::PQ_Simple);
        std::vector<uint32_t> ans = index.search(data[i], topK, 0.9, puffinn::FilterType::None);
        
        std::sort(guess.begin(), guess.end());
        std::sort(ans.begin(), ans.end());
        
        std::vector<uint32_t> v(topK);
        std::vector<uint32_t>::iterator it;
        
        it = std::set_intersection(guess.begin(), guess.end(), ans.begin(), ans.end(),v.begin());
        std::cout << i << "," << (it - v.begin())<< std::endl;
    }

}

void qualityAsProg(){
    std::vector<std::vector<float>> data;
    std::string data_path = "data/glove-25-angular.hdf5";
    auto dims = utils::load(data, "train", data_path, 100000);
    puffinn::Index<puffinn::CosineSimilarity> index(dims.second, 600*1024*1024, false);
    for (std::vector<float> & v : data) { index.insert(v); }
    index.rebuild();
    for(int i = 0; i < 500; i++){
        index.search(data[i*100], 10, 0.95, puffinn::FilterType::PQ_Simple);
    }
    
}

void all_bench()
{
    ankerl::nanobench::Bench bencher;
    bencher.minEpochIterations(2000);
    //correctnessMeassurementPQ();
    imp(&bencher);
    //mahaBench(&bencher);
    //eucBench(&bencher);

    //qualityAsProg();
    //bencher.minEpochIterations(10);
    //bencher.minEpochIterations(10);
    //pqfBench(&bencher);
    //kmeansBench(&bencher);
}