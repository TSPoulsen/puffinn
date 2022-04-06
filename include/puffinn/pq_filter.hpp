#pragma once
#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include "math.h"
#include <vector>
#include <cfloat>
#include <iostream>
namespace puffinn{

    class PQFilter{
        unsigned int M, dims;
        unsigned int K;
        const KMeans::distanceType MODE;
        //codebook that contains m*k centroids
        std::vector<Dataset<UnitVectorFormat>> codebook;
        //precomputed inter-centroid distances for faster symmetric distance computation
        std::vector<std::vector<std::vector<int16_t>>> centroidDistances;
        //pointer to float array of K x M (flattened to 1d array)
        int16_t *queryDistances;
        std::vector<std::vector<uint8_t>> pqCodes;
        Dataset<UnitVectorFormat> &dataset;
        //meta information about the subspaces to avoid recomputation 
        std::vector<unsigned int> subspaceSizes, offsets = {0}, subspaceSizesStored;
        bool is_build = false;
        public:
        
        ///Builds short codes for vectors using Product Quantization by projecting every subspace down to neigherst kmeans cluster
        ///Uses these shortcodes for fast estimation of inner product between vectors  
        ///
        ///@param dataset Reference to the dataset. This enables instantiation of class before data is known
        ///@param m Number of subspaces to split the vectors in, distributes sizes as uniformely as possible
        ///@param k Number of clusters for each subspace, build using kmeans
        ///@param mode The distance meassure that kmeans will use. Currently supportes -> ("euclidian, "mahalanobis")   
        PQFilter(Dataset<UnitVectorFormat> &dataset, unsigned int m = 16, unsigned int k = 256, KMeans::distanceType mode = KMeans::euclidean)
        :M(m),
        dims(dataset.get_description().args),
        K(k),
        MODE(mode),
        dataset(dataset)
        {
            subspaceSizes.resize(M);
            fill(subspaceSizes.begin(), subspaceSizes.end(), dims/M);
            unsigned int leftover = dims - ((dims/M) * M);
            for(auto i = subspaceSizes.begin(); i != subspaceSizes.begin()+leftover; i++) (*i)++;
            auto p = subspaceSizes.begin();
            for(unsigned int i = 1; i < M; i++) offsets.push_back(offsets.back()+ *p++);
            queryDistances = new int16_t[K*M];
        }

        ~PQFilter(){
            delete[] queryDistances;
        }

        //Builds the codebook and computes the inter-centroid-distances
        //Should be called every time the dataset significantly changes a d atleast once before querying
        void rebuild()
        {
            if (dataset.get_size() == 0){
                is_build = false;
                std::fill_n(queryDistances, K*M, 0);
                return;
            } 

            pqCodes.resize(dataset.get_size());
            createCodebook();
            createDistanceTable();
            is_build = true;
        }

        uint64_t memory_usage()
        {
            uint64_t cb_mem = 0u;
            for (Dataset<UnitVectorFormat> &d : codebook){
                cb_mem += d.memory_usage();
            }
            return cb_mem + (pqCodes.size() * M * sizeof(uint8_t));
        }    


    #if __AVX2__        
        //Precompute all distance between centroids using AVX2 instructions
        void createDistanceTable(){
            for(unsigned int m = 0; m < M; m++){
                std::vector<std::vector<int16_t>> subspaceDists;
                for(int k1 = 0; k1 < K; k1++){
                    std::vector<int16_t> dists;
                    for(int k2 = 0; k2 < K; k2++){
                        dists.push_back(dot_product_i16_avx2(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }

    #else        
        //Precompute all distance between centroids 
        void createDistanceTable(){
            for(unsigned int m = 0; m < M; m++){
                std::vector<std::vector<int16_t>> subspaceDists;
                for(int k1 = 0; k1 < K; k1++){
                    std::vector<int16_t> dists;
                    for(int k2 = 0; k2 < K; k2++){
                        dists.push_back(dot_product_i16_simple(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }
    #endif

        //builds a dataset where each vector only contains the mth chunk
        std::vector<std::vector<float>> getSubspace(unsigned int m) {
            unsigned int N = dataset.get_size();
            std::vector<std::vector<float>> subspace(N, std::vector<float>(subspaceSizes[m]));
            for (unsigned int i = 0; i < N; i++) {
                UnitVectorFormat::Type *start = dataset[i] + offsets[m];
                for (unsigned int d = 0; d < subspaceSizes[m]; d++) {
                    subspace[i][d] = UnitVectorFormat::from_16bit_fixed_point(*(start + d));
                    
                }
            }
            return subspace;
        }


        //Runs kmeans for all m subspaces and stores the centroids in codebooks
        void createCodebook(){
            unsigned int k = std::min(K, dataset.get_size());
            //used to keep track of where subspace begins
            for(unsigned int m = 0; m < M ; m++)
            {
                //RunKmeans for the given subspace
                //gb_labels for this subspace will be the mth index of the PQcodes
                
                KMeans kmeans(k, MODE); 
                std::vector<std::vector<float>> subspace = getSubspace(m);
                kmeans.fit(subspace);
                std::vector<std::vector<float>> centroids  = kmeans.getAllCentroids();
                
                //precompute pqCodes for all points in dataset
                for(unsigned int i = 0; i < k; i++){
                    for(unsigned int mem: kmeans.getGBMembers(i)){
                        pqCodes[mem].push_back(i);
                    }
                }

                // Convert back to UnitVectorFormat and store in codebook
                codebook.push_back(Dataset<UnitVectorFormat>(subspaceSizes[m], dataset.get_size()));
                for (unsigned int i = 0; i < k; i++) {
                    UnitVectorFormat::Type *c_p = codebook[m][i];
                    float *vec_p = &centroids[i][0];
                    for (unsigned int d = 0; d < subspaceSizes[m]; d++) {
                        *c_p++ = UnitVectorFormat::to_16bit_fixed_point(*vec_p++);
                    }
                }
                //Sizes of the padded subspaces  
                subspaceSizesStored.push_back(codebook[m].get_description().storage_len);
            }
        }



        //Naive way of getting PQCode will be usefull if we decide to use samples to construct centroids
        std::vector<uint8_t> getPQCode(typename UnitVectorFormat::Type* vec) const {
            std::vector<uint8_t> pqCode;
            for(unsigned int m = 0; m < M; m++){
                float minDistance = FLT_MAX;
                uint8_t quantization = 0;
                for(int k = 0; k < K; k++){
                    float d = UnitVectorFormat::distance(vec+offsets[m], codebook[m][k], subspaceSizes[m]);
                    if(d < minDistance){
                        minDistance = d;
                        quantization = k;    
                    }
                }
                pqCode.push_back(quantization);
            }
            return pqCode;
        }

        void precomp_query_to_centroids(typename UnitVectorFormat::Type* y) const {
            if (!is_build) return;
            int16_t * p = queryDistances;
            for(unsigned int m = 0; m < M; m++){
                for(unsigned int k = 0; k < K; k++){
                    *p++ = asymmetricDistanceComputation_avx(codebook[m][k], y);                     
                }
            }

        }

        //Distance from PQCode to actual vector
        float quantizationError_simple(typename UnitVectorFormat::Type* vec) const {
            float sum = 0;
            std::vector<uint8_t> pqCode = getPQCode(vec);
            int centroidID;
            for(unsigned int m = 0; m < M; m++){
                centroidID = pqCode[m];
                sum += UnitVectorFormat::distance(vec + offsets[m], codebook[m][centroidID], subspaceSizes[m]);
            }
            
            return sum;
        }
        //Function overload to allow idx calls if vector is in the dataset
        float quantizationError_simple(unsigned int idx) const {
            return quantizationError_simple(dataset[idx]);
        }
        
        //Distance from PQCode to actual vector using precomputed PQCodes
        float quantizationError(unsigned int vec_i) const {
            float sum = 0;
            std::vector<uint8_t> pqCode = pqCodes[vec_i];
            int centroidID;
            for(unsigned int m = 0; m < M; m++){
                centroidID = pqCode[m];
                sum += UnitVectorFormat::distance(dataset[vec_i] + offsets[m], codebook[m][centroidID], subspaceSizes[m]);
            }
            return sum;
        }

        float totalQuantizationError_simple() const {
            float sum = 0;
            for(unsigned int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError_simple(dataset[i]);
            }
            return sum;
        }

        float totalQuantizationError() const {
            float sum = 0;
            for(unsigned int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError(i);
            }
            return sum;
        }

        //symmetric distance estimation, PQcodes computed at runtime
        float symmetricDistanceComputation_simple(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y)const {
            float sum = 0;
            //quantize x and y
            std::vector<uint8_t> px = getPQCode(x), py = getPQCode(y);
            //approximate distance by product quantization (precomputed centroid distances required)
            for(unsigned int m = 0; m < M; m++){
                sum += centroidDistances[m][px[m]][py[m]];
            }
            return sum;
        }

        //symmetric distance estimation with precomputed PQcode for x
        int16_t symmetricDistanceComputation(unsigned int xi, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            //quantize x and y
            std::vector<uint8_t> px = pqCodes[xi], py = getPQCode(y);
            //approximate distance by product quantization (precomputed centroid distances required)
            for(unsigned int m = 0; m < M; m++){
                sum += centroidDistances[m][px[m]][py[m]];
            }
            return sum;
        }

        //used to allocate enough memory to pad query point
        unsigned int getPadSize() const {
            unsigned int ans = 0;
            for(unsigned int m = 0; m < M; m++){
                ans += codebook[m].get_description().storage_len;
            }
            return ans;
        }        

        //builds a vector padded to align with each subspace at memory pointed to by "a"
        void createPaddedQueryPoint(typename UnitVectorFormat::Type* y, int16_t *a) const {
            unsigned int tmp[16] = {0};
            for(unsigned int m = 0; m < M; m++){
                for(unsigned int i = 0; i < subspaceSizes[m]; i++){
                    *a++ = *y++;
                }
                unsigned int padd = 16 - (subspaceSizes[m] % 16);
                std::copy_n(tmp, padd, a);
                a += padd;
            }
        }  
        
        //asymmetric distance estimation, PQCode computed at runtime
        int16_t asymmetricDistanceComputation_simple(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            std::vector<uint8_t> px = getPQCode(x);
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_simple(y + offsets[m], codebook[m][px[m]], subspaceSizes[m]);
            }
            return sum;
        }

        #if __AVX2__
        // Fastest version of asymmetric distance computation using avx2
        ///@param xi index of vector in the dataset
        ///@param y pointer to start of UnitVector (padded for each subspace)
        
        int16_t asymmetricDistanceComputation_avx(unsigned int xi, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            const uint8_t *px_p = &pqCodes[xi][0];
            const unsigned int *size_p = &subspaceSizesStored[0];
            const Dataset<UnitVectorFormat> *cb_p = &codebook[0];
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_avx2(y, (*cb_p++)[*px_p++], *size_p);
                y+= *size_p++;
            }
            return sum;
        }
        
        //ovearlead function so you can query with pointer to vector instead of idx in dataset
        int16_t asymmetricDistanceComputation_avx(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y) const {
            int16_t sum = 0;
            const unsigned int *size_p = &subspaceSizesStored[0];
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_avx2(y, x, *size_p);
                x+= *size_p;
                y+= *size_p++;   
            }
            return sum;
        }

        int16_t estimatedInnerProduct(unsigned int xi) const {
            int16_t sum = 0;
            const uint8_t* p = &pqCodes[xi][0];
            for(unsigned int m = 0; m < M; m++){
                sum += queryDistances[*p++ + (m*M)];
            }
            return sum; 
        }
        
        #else
        int16_t asymmetricDistanceComputation_avx(unsigned int xi, typename UnitVectorFormat::Type* y) const {
            std::cerr << "assymetric avx failed -> no AVX2 found" << std::endl;
            return asymmetricDistanceComputation(xi, y);
        }

        #endif

        //asymmetric distance estimation using precomputed PQcodes
        ///@param xi index of vector in the dataset
        ///@param y pointer to start of UnitVector (not padded for each subspace)
        int16_t asymmetricDistanceComputation(unsigned int xi, typename UnitVectorFormat::Type* y) const{
            int16_t sum = 0;
            const uint8_t *px_p = &pqCodes[xi][0];
            const unsigned int *size_p = &subspaceSizes[0];
            const Dataset<UnitVectorFormat> *cb_p = &codebook[0];
            for(unsigned int m = 0; m <M; m++){
                sum += dot_product_i16_simple(y, (*cb_p++)[*px_p++], *size_p);
                y += *size_p++;
            }
            return sum;
        }


        //Functions below are just debugging tools and old code that might be useful down the road
        /*
        std::vector<float> getCentroid(unsigned int mID, unsigned int kID){
            return std::vector<float>(codebook[mID][kID], codebook[mID][kID]+ subspaceSizes[mID]);
        }

        void showCodebook(){
            for(unsigned int m = 0; m < M; m++){
                std::cout << "subspace: " << m << std::endl;
                for(int k = 0; k < K; k++){
                    std::cout << "cluster: "<< k << std::endl;
                    for(unsigned int l = 0; l < codebook[m].get_description().storage_len; l++){
                        std::cout << "\t" <<  UnitVectorFormat::from_16bit_fixed_point(*(codebook[m][k] + l)) << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            } 
        }

        void showPQCodes(){
            std::cout << "PQCODOES1: " << std::endl;
            for(unsigned int i = 0; i < dataset.get_size(); i++){
                for(uint8_t val: getPQCode(dataset[i])){
                    std::cout << (unsigned int) val << " ";
                }
            std::cout << std::endl;
            }
        }

        

        void showSubSizes(){
            for(auto a: subspaceSizes){
                std::cout << a << " ";
            }
            std::cout << std::endl;
        }

        //constructor is depricated
        
        PQFilter(Dataset<UnitVectorFormat> &dataset, std::vector<unsigned int> subs, unsigned int k = 256, KMeans::distanceType mode = KMeans::euclidean)
        :M(subs.size()),
        dims(dims),
        K(k),
        MODE(mode),
        dataset(dataset)
        {   

            pqCodes.resize(dataset.get_size());
            setSubspaceSizes(subs);
            createCodebook();
            createDistanceTable();
        }
        //helper function for depricated constructor
        void setSubspaceSizes(std::vector<unsigned int> subs){
            subspaceSizes = subs;
            offsets.clear();
            for(unsigned int i = 1; i < M; i++) offsets[i] = subs[i-1] + offsets[i-1]; 
        }
        */
    };
}