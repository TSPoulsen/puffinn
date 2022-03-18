#pragma once
#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include <vector>
#include <cfloat>
#include <iostream>
namespace puffinn{
    class PQFilter{
        unsigned int M, dims;
        unsigned int K;
        KMeans::distanceType MODE;
        //codebook that contains m*k centroids
        std::vector<Dataset<UnitVectorFormat>> codebook;
        std::vector<std::vector<std::vector<float>>> centroidDistances;
        std::vector<std::vector<uint8_t>> pqCodes;
        Dataset<UnitVectorFormat> &dataset;
        std::vector<unsigned int> subspaceSizes, offsets = {0};
        public:
        PQFilter(Dataset<UnitVectorFormat> &dataset, unsigned int m = 16, unsigned int k = 256, KMeans::distanceType mode = KMeans::euclidean)
        :M(m),
        dims(dataset.get_description().args),
        K(k),
        MODE(mode),
        dataset(dataset)
        {
            subspaceSizes.resize(M);
            pqCodes.resize(dataset.get_size());
            fill(subspaceSizes.begin(), subspaceSizes.end(), dims/M);
            unsigned int leftover = dims - ((dims/M) * M);
            for(unsigned int i = 0; i < leftover; i++) subspaceSizes[i]++;
            for(unsigned int i = 1; i < M; i++) offsets.push_back(offsets.back()+ subspaceSizes[i-1]);
        }


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
         
        void rebuild()
        {
            createCodebook();
            createDistanceTable();
        }

        void setSubspaceSizes(std::vector<unsigned int> subs){
            subspaceSizes = subs;
            offsets.clear();
            for(unsigned int i = 1; i < M; i++) offsets[i] = subs[i-1] + offsets[i-1]; 
        }

        //Precompute all distance between centroids 
    #if __AVX__        
        void createDistanceTable(){
            for(unsigned int m = 0; m < M; m++){
                std::vector<std::vector<float>> subspaceDists;
                for(int k1 = 0; k1 < K; k1++){
                    std::vector<float> dists;
                    for(int k2 = 0; k2 < K; k2++){
                        dists.push_back(UnitVectorFormat::innerProduct_avx(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }

    #else        
        void createDistanceTable(){
            for(unsigned int m = 0; m < M; m++){
                std::vector<std::vector<float>> subspaceDists;
                for(int k1 = 0; k1 < K; k1++){
                    std::vector<float> dists;
                    for(int k2 = 0; k2 < K; k2++){
                        dists.push_back(UnitVectorFormat::innerProduct(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }
    #endif


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
            //used to keep track of where subspace begins
            for(unsigned int m = 0; m < M ; m++)
            {
                //RunKmeans for the given subspace
                //gb_labels for this subspace will be the mth index of the PQcodes
                
                KMeans kmeans(K, MODE); 
                std::vector<std::vector<float>> subspace = getSubspace(m);
                kmeans.fit(subspace);
                std::vector<std::vector<float>> centroids  = kmeans.getAllCentroids();
                //precompute pqCodes for all points in dataset
                for(unsigned int i = 0; i < K; i++){
                    for(unsigned int mem: kmeans.getGBMembers(i)){
                        pqCodes[mem].push_back(i);
                    }
                }

                // Convert back to UnitVectorFormat and store in codebook
                codebook.push_back(Dataset<UnitVectorFormat>(subspaceSizes[m], dataset.get_size()));
                for (unsigned int i = 0; i < K; i++) {
                    UnitVectorFormat::Type *c_p = codebook[m][i];
                    float *vec_p = &centroids[i][0];
                    for (unsigned int d = 0; d < subspaceSizes[m]; d++) {
                        *c_p++ = UnitVectorFormat::to_16bit_fixed_point(*vec_p++);
                    }
                }
                
            }
            //showCodebook();
            //showPQCodes();
            //std::cout << "Calculating quantization error for index 1: " << quantizationError(1) << std::endl;
        }


        // Runtime of thisdataset[i] is K*dims
        std::vector<uint8_t> getPQCode(typename UnitVectorFormat::Type* vec){
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



        float quantizationError(typename UnitVectorFormat::Type* vec){
            float sum = 0;
            std::vector<uint8_t> pqCode = getPQCode(vec);
            int centroidID;
            for(unsigned int m = 0; m < M; m++){
                centroidID = pqCode[m];
                sum += UnitVectorFormat::distance(vec + offsets[m], codebook[m][centroidID], subspaceSizes[m]);
            }
            /*
            std::cout <<" quantization error for: " << index << " ";
            for(int k = 0; k < dims; k++){
                std::cout << vec[k] << " ";  
            }
            std::cout << sum << std::endl;
            */
            return sum;
        }
        float quantizationError_fast(unsigned int vec_i){
            float sum = 0;
            std::vector<uint8_t> pqCode = pqCodes[vec_i];
            int centroidID;
            for(unsigned int m = 0; m < M; m++){
                centroidID = pqCode[m];
                sum += UnitVectorFormat::distance(dataset[vec_i] + offsets[m], codebook[m][centroidID], subspaceSizes[m]);
            }
            return sum;
        }




        float quantizationError(unsigned int idx) {
            return quantizationError(dataset[idx]);
        }

        float totalQuantizationError(){
            float sum = 0;
            for(unsigned int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError(dataset[i]);
            }
            return sum;
        }

        float totalQuantizationError_fast(){
            float sum = 0;
            for(unsigned int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError_fast(i);
            }
            return sum;
        }

        float symmetricDistanceComputation(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y){
            float sum = 0;
            //quantize x and y
            std::vector<uint8_t> px = getPQCode(x), py = getPQCode(y);
            //approximate distance by product quantization (precomputed centroid distances required)
            for(unsigned int m = 0; m < M; m++){
                sum += centroidDistances[m][px[m]][py[m]];
            }
            return sum;
        }
        
        float symmetricDistanceComputation_fast(unsigned int xi, typename UnitVectorFormat::Type* y){
            float sum = 0;
            //quantize x and y
            std::vector<uint8_t> px = pqCodes[xi], py = getPQCode(y);
            //approximate distance by product quantization (precomputed centroid distances required)
            for(unsigned int m = 0; m < M; m++){
                sum += centroidDistances[m][px[m]][py[m]];
            }
            return sum;
        }

        unsigned int getPadSize(){
            unsigned int ans = 0;
            for(unsigned int m = 0; m < M; m++){
                ans += codebook[m].get_description().storage_len;
                //std::cout << "size of a subspace hue: " << codebook[m].get_description().storage_len << std::endl;
            }
            return ans;
        }        

        //builds a vector padded to align with each subspace at memory pointed to by a
        void createPaddedQueryPoint(typename UnitVectorFormat::Type* y, int16_t *a){
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
        
        float asymmetricDistanceComputation(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y){
            float sum = 0;
            std::vector<uint8_t> px = getPQCode(x);
            for(unsigned int m = 0; m <M; m++){
                sum += UnitVectorFormat::innerProduct(y + offsets[m], codebook[m][px[m]], subspaceSizes[m]);
            }
            return sum;
        }
        #if __AVX__
        float asymmetricDistanceComputation_fast_avx(unsigned int xi, typename UnitVectorFormat::Type* y){
            float sum = 0;
            std::vector<uint8_t> px = pqCodes[xi];
            for(unsigned int m = 0; m <M; m++){
                sum += UnitVectorFormat::innerProduct_avx(y, codebook[m][px[m]], codebook[m].get_description().storage_len);
                y+= codebook[m].get_description().storage_len;
            }
            return sum;
        }
        #endif

        float asymmetricDistanceComputation_fast(unsigned int xi, typename UnitVectorFormat::Type* y){
            float sum = 0;
            std::vector<uint8_t> px = pqCodes[xi];
            for(unsigned int m = 0; m <M; m++){
                sum += UnitVectorFormat::innerProduct(y + offsets[m], codebook[m][px[m]], subspaceSizes[m]);
            }
            return sum;
        }

        std::vector<float> getCentroid(unsigned int mID, unsigned int kID){
            return std::vector<float>(codebook[mID][kID], codebook[mID][kID]+ subspaceSizes[mID]);
        }

        //Functions below are just debugging tools 
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
    };
}