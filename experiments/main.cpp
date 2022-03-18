
//#include "l2_quantization_error.hpp"
//#include "ang_dist.hpp"
//#include "time_kmeans.hpp"
#include <nanobench.h>
#include "puffinn/pq_filter.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "PQFilter_bench.hpp"
#include "set"
#include "benchmarks.hpp"
using namespace puffinn;

int main() {
    // ang_dist_glove_subset();
    // time_kmeans();
    // l2_quant_error();
    all_bench();
}
