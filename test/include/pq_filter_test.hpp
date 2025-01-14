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
        pq1.rebuild();
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
        pq1.rebuild();
        //pq1.showCodebook();
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
        pq1.rebuild(); 

        //Since cluster 0 might not have the same values each time
        //we have to use the quantization error to see if we are generating the correct PQcodes
        REQUIRE(0.0 != pq1.totalQuantizationError());
    }
    
    TEST_CASE("precomp test") {
        unsigned int N = 8, dims = 4, m = 2, k  = 4;
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
        pq1.rebuild();
        REQUIRE(pq1.totalQuantizationError() == pq1.totalQuantizationError_simple());
        
        alignas(32) int16_t tmp[pq1.getPadSize()];
        pq1.createPaddedQueryPoint(dataset[1], tmp);

        REQUIRE(pq1.asymmetricDistanceComputation_simple(dataset[0], dataset[1]) == pq1.asymmetricDistanceComputation(0, dataset[1]));
        REQUIRE(pq1.symmetricDistanceComputation_simple(dataset[2], dataset[3]) == pq1.symmetricDistanceComputation(2, dataset[3]));
    }
    
    TEST_CASE("AsymmetricFast Test"){
        std::vector<float> input[30] = {
            {-1.82445727,  0.04013046},
            { 0.28121734, -0.25688856},
            {-1.82976968,  0.02095528},
            { 0.83968045,  0.30968837},
            { 0.91359481, -0.30797411},
            {-0.81865681,  0.4669873 },
            { 0.07045335, -0.48246012},
            { 1.97495704, -0.00925621},
            {-0.64461239, -0.1055485 },
            { 1.67525065,  0.01462244},
            { 1.17554273,  1.04013046},
            { 3.28121734,  0.74311144},
            { 1.17023032,  1.02095528},
            { 3.83968045,  1.30968837},
            { 3.91359481,  0.69202589},
            { 2.18134319,  1.4669873 },
            { 3.07045335,  0.51753988},
            { 4.97495704,  0.99074379},
            { 2.35538761,  0.8944515 },
            { 4.67525065,  1.01462244},
            { 4.17554273,  0.04013046},
            { 6.28121734, -0.25688856},
            { 4.17023032,  0.02095528},
            { 6.83968045,  0.30968837},
            { 6.91359481, -0.30797411},
            { 5.18134319,  0.4669873 },
            { 6.07045335, -0.48246012},
            { 7.97495704, -0.00925621},
            { 5.35538761, -0.1055485 },
            { 7.67525065,  0.01462244}};

        // Needs more runs and iterations for the result to be stable
        
        unsigned int N = 30, dims = 2, m = 1, k  = 4;
        Dataset<UnitVectorFormat> dataset(dims, N);
        for (auto entry: input){
            dataset.insert(entry);
        }
        PQFilter pq1(dataset, m,k);
        pq1.rebuild();
        bool isSame = true;
        for(unsigned int i = 0; i < N; i++){
            for( unsigned int j = 0; j < N; j++){
                alignas(32) int16_t tmp[pq1.getPadSize()];
                pq1.createPaddedQueryPoint(dataset[j], tmp);
                float hue = pq1.asymmetricDistanceComputation_simple(dataset[i], dataset[j]);
                float hue1 = pq1.asymmetricDistanceComputation_avx(i, tmp);
                isSame = (hue == hue1);
            }          
        }
        REQUIRE(isSame);
        
    }
    


}
