#pragma once
#include "puffinn/dataset.hpp"
#include "puffinn/format/unit_vector.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/math.hpp"

#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <array>
#include <random>
#include <unordered_set>
#include <cfloat>

#if defined(__AVX2__) || defined(__AVX__)
    #include <immintrin.h>
#endif

namespace puffinn
{
    /// Class for performing k-means clustering on a given dataset
    class KMeans
    {
    public:
        struct Cluster {
            std::vector<float> centroid;
            std::vector<unsigned int> members;
            Cluster(){}
            Cluster(const Cluster &c1){
                centroid = c1.centroid;
                members = c1.members;
            }
        };
        enum distanceType {euclidean, mahalanobis, none};
    private:

        using dataType = std::vector<std::vector<float>>;

        // Clustering configuration
        unsigned int padding = 0;
        const unsigned int K;
        const float TOL;
        const uint16_t MAX_ITER;
        const unsigned int N_RUNS;
        std::vector<float> covarianceMatrix;
        distanceType MODE;

        // gb_centroids is for the global best set of centroids
        std::vector<Cluster> gb_clusters;
        float gb_inertia = FLT_MAX;
        unsigned int padding = 0;

    public:
        KMeans(unsigned int K_clusters = 256, distanceType mode = euclidean, unsigned int runs = 3, unsigned int max_iter = 100, float tol = 0.0001f)
            : K(K_clusters),
            TOL(tol),
            MAX_ITER(max_iter),
            N_RUNS(runs),
            MODE(mode)
        {
            assert(K <= 256);
            std::cerr << "Kmeans info: tK=" << (unsigned int)K << std::endl;
        }

        ~KMeans() {}

        void fit(dataType &data)
        {
            // add padding to make each vector a multiple of 8
            padData(data);

            if (MODE == mahalanobis){
                createCovarianceMatrix(data);
            }

            //clean up for next subspace fit
            gb_inertia = FLT_MAX;
            // init_centers_random(); doesn't work
            for(unsigned int run=0; run < N_RUNS; run++) {
                std::cerr << "Run " << run+1 << "/" << N_RUNS << std::endl;
                std::vector<Cluster> clusters = init_centroids_kpp(data);
                float run_inertia = lloyd(data, clusters);
                if (run_inertia < gb_inertia) {
                    //New run is the currently best, thus overwrite gb variables
                    std::cout << "assigning new gb_clusters" << std::endl;
                    gb_inertia = run_inertia;
                    gb_clusters =  clusters; // Copies the whole Class 
                }
            }


        }

        std::vector<float> getCentroid(size_t c_i) {
            // Removes padding from centroid
            return std::vector<float>(&*gb_clusters[c_i].centroid.begin(), &*gb_clusters[c_i].centroid.end()-padding);
        }

        dataType getAllCentroids(){
            dataType all_centroids(K);
            for (unsigned int c_i = 0; c_i < K; c_i++){
                all_centroids[c_i] = getCentroid(c_i);
            }
            return all_centroids;
        }

        std::vector<float> getCovarianceMatrix(){
            return covarianceMatrix;
        }

        //pointers to the actual start should be given i.e. offset should be handled outside this function
        double distance(std::vector<float> &v1, std::vector<float> &v2){

            if(MODE == euclidean){
                return sumOfSquares(v1, v2);
            }
            
            if(MODE == mahalanobis){
                return mahaDistance(v1,v2);
            }
        }

        double mahaDistance(std::vector<float> &v1, std::vector<float> &v2)
        {
                //std::cerr << "Maha distance being computed" << std::endl;
                std::vector<float> delta(v1.size());
                for(unsigned int d = 0; d < v1.size(); d++){
                    delta[d] = v1[d] - v2[d];
                }

                // Use SIMD HERE
                std::vector<float> temp(v1.size(), 0);
                for(unsigned int d = 0; d < v1.size(); d++){
                    for(unsigned int delta_i = 0; delta_i < v1.size(); delta_i++){
                        temp[d] += delta[delta_i] * covarianceMatrix[(v1.size() * delta_i) + d]; 
                    }    
                }

                double distance = 0.0;
                for(unsigned int delta_i = 0; delta_i < v1.size(); delta_i++){
                    distance += ((double) delta[delta_i]) * ((double) temp[delta_i]);
                }
                return distance;                


        }

        double totalError(dataType &data, distanceType mode = none) {
            padData(data);
            if (mode == none) {
                mode = MODE;
            }
            //double (KMeans::* d_ptr)(std::vector<float>&, std::vector<float>&);
            //if (mode == euclidean) d_ptr = &KMeans::sumOfSquares;
            //if (mode == mahalanobis) d_ptr = &KMeans::mahaDistance;
            double total_err = 0;
            for (Cluster &c : gb_clusters) {
                for (unsigned int idx : c.members) {
                    if (mode == euclidean)        total_err += sumOfSquares(data[idx], c.centroid);
                    else if (mode == mahalanobis) total_err += mahaDistance(data[idx], c.centroid);
                    //total_err += (*d_ptr)(data[idx], c.centroid);
                }
                std::cerr << std::endl;
            }
            std::cerr << std::endl << std::endl;
            return total_err;
            

        }

        void createCovarianceMatrix(dataType &data){
            covarianceMatrix.resize(data[0].size()*data[0].size());
            // Use SIMD

            //build covariance matrix by covariance[x][y] = avg((xi * yi))
            //std::cerr << "covariance matrix addition part begin" << std::endl;
            for(unsigned int i = 0;  i < data.size(); i++){
                for(unsigned int d1= 0; d1 < data[0].size(); d1++){
                    for(unsigned int d2 = 0; d2 < data[0].size(); d2++){  
                        covarianceMatrix[(data[0].size() * d1) + d2] += data[i][d1] * data[i][d2];
                    }
                }
                //std::cerr << "finished with vector: " << i << std::endl;
            }
            //std::cerr  << "covariance matrix addition part finished" << std::endl;
            //get average by deviding by n 
            for(unsigned int cov = 0; cov < (data[0].size()*data[0].size()); cov++){
                covarianceMatrix[cov] /= (data.size());
            }
            //std::cerr << "division of covariance matrix finished" << std::endl;
        }



        // Performs a single kmeans clustering using the lloyd algorithm
        float lloyd(dataType &data, std::vector<Cluster> &clusters) 
        {
            double inertia_delta = DBL_MAX;
            double inertia = DBL_MAX;
            unsigned int iteration = 0;

            while (inertia_delta > TOL && iteration < MAX_ITER )
            {
                //std::cerr << "\tlloyd iteration: " << iteration;
                double current_inertia = assignToClusters(data, clusters);
                for (auto cit = clusters.begin(); cit != clusters.end(); cit++) {
                    setCentroidMean(data, *cit);
                }

                inertia_delta = (inertia - current_inertia);
                inertia = current_inertia;
                iteration++;
            }
            return inertia;

        }

    #ifdef __AVX__
        // Adds padding such that each vector is a multiple of 8
        void padData(dataType &data) {
            padding = 8 - (data[0].size() % 8);
            if (padding == 8) return; // Already correct size
            for (std::vector<float> &vec : data) {
                for (unsigned int p = 0; p < padding; p++) {
                    vec.push_back(0);
                }
            }

        }
    #else
        void padData(dataType &data) {} // only pad data when using avx instructions
    #endif

        // samples K random points and uses those as starting centers
        std::vector<Cluster> init_centroids_random(dataType &data)
        {
            std::vector<Cluster> clusters(K);
            std::cerr << "Init random centers" << std::endl;
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, data.size()-1);

            unsigned int c_i = 0;
            std::unordered_set<unsigned int> used;
            std::cerr << "BLYat" << std::endl;
            while(c_i < K) {
                std::cerr << "k" << std::endl;
                unsigned int sample_idx = random_idx(rand_gen);
                if (used.find(sample_idx) == used.end()) {
                    used.insert(sample_idx);
                    clusters[c_i++].centroid = data[sample_idx];
                }
            }
            return clusters;

        }

    // sumOfSquares heavily inspired from https://github.com/yahoojapan/NGT/blob/master/lib/NGT/Clustering.h
    #ifdef __AVX2__
        double sumOfSquares(std::vector<float> &v1, std::vector<float> &v2)
        {
            __m256 sum = _mm256_setzero_ps();
            float *a = &v1[0];
            float *b = &v2[0];
            float *a_end = &v1[v1.size()];
            while (a != a_end) {
                __m256 v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(v, v));
                a += 8;
                b += 8;
            }

            __attribute__((aligned(32))) float f[8];
            _mm256_store_ps(f, sum);
            double s = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
            return s;
        }
    #else
        double sumOfSquares(std::vector<float> &v1, std::vector<float> &v2)
        {

            double csum = 0.0;
            float *a = &v1[0];
            float *b = &v2[0];
            float *a_end = &v1[v1.size()];
            while (a != a_end) {
                double d = (double)*a++ - (double)*b++;
                csum += d * d;
            }
            return csum;
        }
    #endif

        // Kmeans++ initialization of centroids
        std::vector<Cluster> init_centroids_kpp(dataType &data)
        {
            std::vector<Cluster> clusters(K);
            // Pick first centroids uniformly
            auto &rand_gen = get_default_random_generator();
            std::uniform_int_distribution<unsigned int> random_idx(0, data.size()-1);
            clusters[0].centroid = data[random_idx(rand_gen)];

            // Choose the rest proportional to their distance to already chosen centroids
            std::vector<double> distances(data.size(), DBL_MAX);
            for (unsigned int c_i = 1; c_i < K; c_i++) {
                //calculate distances to last chosen centroid
                for (unsigned int i = 0; i < data.size(); i++) {
                    double new_dist = sumOfSquares(data[i], clusters[c_i-1].centroid);
                    distances[i] = std::min(distances[i], new_dist);
                }

                std::discrete_distribution<int> rng(distances.begin(), distances.end());
                clusters[c_i].centroid = data[rng(rand_gen)];
            }
            return clusters;
        }


        // Sets the labels for all vectors, and returns inertia
        double assignToClusters(dataType &data, std::vector<Cluster> &clusters)
        {
            // Clear member variable for each cluster
            for (auto cit = clusters.begin(); cit != clusters.end(); cit++) {
                (*cit).members.clear();
            }

            double inertia = 0;

            for (unsigned int i = 0; i < data.size(); i++) {
                double min_dist = DBL_MAX;
                unsigned int min_label = data.size() + 1;
                for (unsigned int c_i = 0; c_i < K; c_i++) {
                    double d = distance(data[i], clusters[c_i].centroid);
                    if (d < min_dist) {
                        min_label = c_i;
                        min_dist = d;
                    }
                }
                clusters[min_label].members.push_back(i);
                inertia += min_dist;
            }
            // Handle emtpy cluster
            return inertia;
        }

    #ifdef __AVX__
        void setCentroidMean(dataType &data, Cluster &c)
        {
            unsigned int n256 = data[0].size()/8;
            __m256 sum[n256];
            for (unsigned int n = 0; n < n256; n++) {
                sum[n] = _mm256_setzero_ps();
            }
            for (unsigned int idx : c.members) {
                float *a = &data[idx][0];
                for (unsigned int n = 0; n < n256; n++, a+=8) {
                    sum[n] = _mm256_add_ps(sum[n], _mm256_loadu_ps(a));
                }
            }
            assert(c.members.size() != 0);
            float div = 1.0/c.members.size();
            alignas(32) float div_a[8] = {div, div, div, div, div, div, div, div};
            __m256 div_v = _mm256_load_ps(div_a);
            float *cen_s = &c.centroid[0];
            for (unsigned int n = 0; n < n256; n++, cen_s += 8) {
                _mm256_storeu_ps(cen_s,
                    _mm256_mul_ps(sum[n], div_v));
            }
        }
    #else
        void setCentroidMean(dataType &data, Cluster &c)
        {

            fill(c.centroid.begin(), c.centroid.end(), 0.0f);
            for (unsigned int idx : c.members) {
                for (unsigned int d = 0; d < c.centroid.size(); d++) {
                    c.centroid[d] += data[idx][d];
                }
            }
            assert(c.members.size() != 0);
            for (unsigned int d = 0; d < c.centroid.size(); d++) {
                c.centroid[d] /= c.members.size();
            }

        }
    #endif
    };
}