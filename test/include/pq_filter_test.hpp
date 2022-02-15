#pragma once
#include "catch.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/format/real_vector.hpp"
#include "set"
namespace pq{
    using namespace puffinn;

    TEST_CASE("Able to set different sizes for subspaces") {
        unsigned int N = 8, dims = 4, m = 2, k  = 2;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, 7.0, 1.0},
                                        {-3.0, 1.0, 7.0, 1.0},
                                        {-2.0, 1.0, 7.0, 1.0},
                                        {-1.0, 1.0, 7.0, 1.0},
                                        {1.0 , 1.0, 7.0, 8.0},
                                        {2.0 , 1.0, 7.0, 8.0},
                                        {3.0 , 1.0, 7.0, 8.0},
                                        {4.0 , 1.0, 7.0, 8.0}};

        Dataset<RealVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter<RealVectorFormat> pq1(dataset,dims, m, k);
        std::vector<unsigned int> sizes = {1,3};
        pq1.setSubspaceSizes(sizes);
        pq1.createCodebook();
        std::vector<float> c1 = pq1.getCluser(0,0),
                           c2 = pq1.getCluser(0,1), 
                           c3 = pq1.getCluser(1,0), 
                           c4 = pq1.getCluser(1,1);
        
        std::set<std::vector<float>> ans = {c1,c2,c3,c4},
                                    corrct = {
                                        {2.5},
                                        {-2.5},
                                        {1,7,8},
                                        {1,7,1}
                                    };
        REQUIRE(ans == corrct);
      }

    TEST_CASE("Test if default PQFilter works") {
        unsigned int N = 8, dims = 4, m = 2, k  = 2;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, 7.0, 1.0},
                                        {-3.0, 1.0, 7.0, 1.0},
                                        {-2.0, 1.0, 7.0, 1.0},
                                        {-1.0, 1.0, 7.0, 1.0},
                                        {1.0 , 1.0, 7.0, 8.0},
                                        {2.0 , 1.0, 7.0, 8.0},
                                        {3.0 , 1.0, 7.0, 8.0},
                                        {4.0 , 1.0, 7.0, 8.0}};

        Dataset<RealVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter<RealVectorFormat> pq1(dataset,dims, m,k);
        pq1.createCodebook();
        std::vector<float> c1 = pq1.getCluser(0,0),
                           c2 = pq1.getCluser(0,1), 
                           c3 = pq1.getCluser(1,0), 
                           c4 = pq1.getCluser(1,1);
        std::set<std::vector<float>> ans = {c1,c2,c3,c4},
                                    corrct = {
                                        {2.5,1},
                                        {-2.5,1},
                                        {7,8},
                                        {7,1}
                                    };
        
        REQUIRE(ans == corrct);


    }
    TEST_CASE("PQFilter generate correct PQcodes") {
        unsigned int N = 8, dims = 4, m = 4, k  = 8;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, 7.0, 1.0},
                                        {-3.0, 1.0, 7.0, 1.0},
                                        {-2.0, 1.0, 7.0, 1.0},
                                        {-1.0, 1.0, 7.0, 1.0},
                                        {1.0 , 1.0, 7.0, 8.0},
                                        {2.0 , 1.0, 7.0, 8.0},
                                        {3.0 , 1.0, 7.0, 8.0},
                                        {4.0 , 1.0, 7.0, 8.0}};

        Dataset<RealVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter<RealVectorFormat> pq1(dataset,dims, m,k);
        pq1.createCodebook();
        //Since cluster 0 might not have the same values each time
        //we have to use the quantization error to see if we are generating the correct PQcodes
        REQUIRE(0.0 == pq1.totalQuantizationError());

    }
}
