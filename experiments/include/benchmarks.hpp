#include <nanobench.h>
#include <puffinn/kmeans.hpp>
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
    puffinn::PQFilter pq1(ds, 2, 32);
    alignas(32) int16_t tmp[pq1.getPadSize()];
    pq1.createPaddedQueryPoint(ds[110], tmp);

    bencher->run("Asymmetric computing PQ code every call", [&] {
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation(ds[0], ds[110]));
    });    
    
    bencher->run("Asymmetric PQ code precomputed", [&] {
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation_fast(0, ds[110]));
    });    
    
    bencher->run("Asymmetric fast creating padded querry once", [&] {
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation_fast_avx(0, tmp));
    });

    bencher->run("Asymmetric fast creating padded querry before each call", [&] {
        alignas(32) int16_t tmp1[pq1.getPadSize()];
        pq1.createPaddedQueryPoint(ds[110], tmp1);
        ankerl::nanobench::doNotOptimizeAway(pq1.asymmetricDistanceComputation_fast_avx(0, tmp1));
    });    

}

void all_bench()
{
    ankerl::nanobench::Bench bencher;
    bencher.minEpochIterations(5000);
    mahaBench(&bencher);
    eucBench(&bencher);
    bencher.minEpochIterations(10);
    bencher.minEpochIterations(10);
    pqfBench(&bencher);
    kmeansBench(&bencher);
}