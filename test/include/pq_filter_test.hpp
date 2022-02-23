#pragma once
#include "catch.hpp"
#include "puffinn/pq_filter.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "set"
namespace pq{
    using namespace puffinn;

    TEST_CASE("PQFilter generate correct PQcodes") {
        unsigned int N = 8, dims = 4, m = 4, k  = 8;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, -8.0, 1.0},
                                        {-3.0, 2.0, -7.0, 2.0},
                                        {-2.0, 3.0, -6.0, 3.0},
                                        {-1.0, 4.0, -5.0, 4.0},
                                        {1.0 , 5.0, -4.0, 5.0},
                                        {2.0 , 6.0, -3.0, 6.0},
                                        {3.0 , 7.0, -2.0, 7.0},
                                        {4.0 , 8.0, -1.0, 8.0}};

        Dataset<UnitVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter pq1(dataset, m,k);
        pq1.createCodebook();
        //Since cluster 0 might not have the same values each time
        //we have to use the quantization error to see if we are generating the correct PQcodes
        REQUIRE(0.0 == pq1.totalQuantizationError());
    }

    TEST_CASE("PQFilter generate correct PQcodes 2") {
        unsigned int N = 4, dims = 4, m = 2, k = 2;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, -8.0, 1.0},
                                        {-4.0, 1.0,  8.0,-1.0},
                                        { 4.0,-1.0, -8.0, 1.0},
                                        { 4.0,-1.0,  8.0,-1.0}};

        Dataset<UnitVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter pq1(dataset, m,k);
        pq1.createCodebook();
        //Since cluster 0 might not have the same values each time
        //we have to use the quantization error to see if we are generating the correct PQcodes
        REQUIRE(0.0 == pq1.totalQuantizationError());
    }

    TEST_CASE("PQFilter generate PQcodes with some Quantization Error") {
        unsigned int N = 6, dims = 4, m = 2, k = 2;
        std::vector<float>  data[N] = {
                                        {-4.0, 1.0, -8.0, 1.0},
                                        {-4.0, 1.0,  8.0,-1.0},
                                        { 4.0,-1.0, -8.0, 1.0},
                                        { 4.0,-1.0,  8.0,-1.0},
                                        { 1.0, 0.0,  1.0, 0.0},
                                        { 0.0,-1.0,  0.0,-1.0},
                                        };

        Dataset<UnitVectorFormat> dataset(dims, N);
        for (auto entry: data){
            dataset.insert(entry);
        }
        PQFilter pq1(dataset, m,k);
        pq1.createCodebook();
        //Since cluster 0 might not have the same values each time
        //we have to use the quantization error to see if we are generating the correct PQcodes
        REQUIRE(0.0 != pq1.totalQuantizationError());
    }

}
