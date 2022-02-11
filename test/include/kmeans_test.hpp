#pragma once

#include "catch.hpp"
#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include "puffinn/format/real_vector.hpp"

#include <vector>
#include <iostream>
#include <set>

using namespace puffinn;
namespace kmeans {
    struct TestData {
        unsigned int N, dims, K;
        std::vector<std::vector<float>> input;
        std::set<std::vector<float>> answer;
    };

    bool are_approx_equal(std::set<std::vector<float>> run_answer, std::set<std::vector<float>> real_answer,float margin = 0.001)
    {
        if (run_answer.size() != real_answer.size()) return false;
        size_t K  = run_answer.size();
        size_t dims  = (*run_answer.begin()).size();
        
        auto it_run = run_answer.begin(), it_real = real_answer.begin();
        bool are_equal = true;
        for (size_t i = 0; i < K; i++) {
            std::vector<float> run_i = *it_run, real_i = *it_real;
            if (run_i.size() != dims || real_i.size() != dims) return false;
            for (size_t d = 0; d < dims; d++) {
                are_equal = are_equal && (run_i[d] == Approx(real_i[d]).margin(margin));
            }
            it_run++;
            it_real++;
        }
        return are_equal;

    }

    void kmeans_correctness_test(struct TestData td)
    {

        Dataset<UnitVectorFormat> dataset(td.dims, td.N);
        for (auto entry : td.input) {
            dataset.insert(entry);
        }
        KMeans kmeans(dataset, (uint8_t)td.K);
        kmeans.fit();
        std::set<std::vector<float>> run_ans;
        for (unsigned int i = 0; i < td.K; i++) {
            UnitVectorFormat::Type* cen = kmeans.getCentroid(i);
            std::vector<float> float_cen(td.dims, 0);
            for (unsigned int d = 0; d < td.dims; d++) {
                float_cen[d] = UnitVectorFormat::from_16bit_fixed_point(cen[d]);
            }
            run_ans.insert(float_cen);
        }
        bool are_equal = are_approx_equal(run_ans, td.answer);
        REQUIRE(are_equal);

    }

    TEST_CASE("basic kmeans clustering 1") {

        struct TestData td;
        td.N    = 4;
        td.dims = 2;
        td.K    = 2;
        td.input = {
            {-1.0,0.01},
            {-1.0,-0.01},
            {1.0,0.01},
            {1.0,-0.01}};
        td.answer = {
            {-1.0,0.0},
            {1.0,0.0}
        };
        kmeans_correctness_test(td);
        return;
        
    }

    TEST_CASE("basic kmeans clustering 2") {

        struct TestData td;
        td.N    = 8;
        td.dims = 2;
        td.K    = 4;
        td.input = {
            {-1.0,0.01},
            {-1.0,-0.01},
            {1.0,0.01},
            {1.0,-0.01},

            {0.01,-1.00},
            {-0.01,-1.00},
            {0.01,1.00},
            {-0.01,1.00},
            
            
            
            };
        td.answer = {
            {-1.0,0.0},
            {1.0,0.0},
            {0.0,1.0},
            {0.0,-1.0}
        };
        kmeans_correctness_test(td);
        return;
        
    }

}