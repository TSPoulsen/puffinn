
//#include "l2_quantization_error.hpp"
#include "ang_dist.hpp"
#include "threshhold.hpp"
#include "inertia_variance.hpp"
#include <nanobench.h>
#include "puffinn/pq_filter.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "set"
#include "benchmarks.hpp"
using namespace puffinn;

int main(int argc, char *argv[]) {
    //ang_dist_glove_subset();
    //pq_passing_filter();
    // time_kmeans();
    // l2_quant_error();
    // all_bench();
    //lsh_passing_filter();
    run_pass_filter(argc, argv);
    // inertia_run();
    //inertia_variance();
}
