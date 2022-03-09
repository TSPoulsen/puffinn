#pragma once
#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include <vector>
#include <cfloat>
#include <iostream>
namespace puffinn{
    class PQFilter{
        unsigned int M, dims;
        unsigned char K;
        KMeans::distanceType MODE;
        //codebook that contains m*k centroids
        std::vector<Dataset<UnitVectorFormat>> codebook;
        std::vector<std::vector<std::vector<float>>> centroidDistances;
        Dataset<UnitVectorFormat> &dataset;
        std::vector<unsigned int> subspaceSizes, offsets = {0};
        public:
        PQFilter(Dataset<UnitVectorFormat> &dataset, KMeans::distanceType mode = KMeans::euclidean, unsigned int m = 16, unsigned int k = 256)
        :M(m),
        dims(dataset.get_description().args),
        K(k),
        MODE(mode),
        dataset(dataset)
        {
            subspaceSizes.resize(M);
            fill(subspaceSizes.begin(), subspaceSizes.end(), dims/M);
            unsigned int leftover = dims - ((dims/M) * M);
            for(unsigned int i = 0; i < leftover; i++) subspaceSizes[i]++;
            for(unsigned int i = 1; i < M; i++) offsets.push_back(offsets.back()+ subspaceSizes[i-1]);
            createCodebook();
            createDistanceTable();
        }
        PQFilter(Dataset<UnitVectorFormat> &dataset, KMeans::distanceType mode = KMeans::euclidean, std::vector<unsigned int> subs, unsigned int k = 256)
        :M(subs.size()),
        dims(dims),
        K(k),
        MODE(mode),
        dataset(dataset)
        {   
            setSubspaceSizes(subs);
            createCodebook();
            createDistanceTable();
        }
        
        
        void setSubspaceSizes(std::vector<unsigned int> subs){
            subspaceSizes = subs;
            offsets.clear();
            for(unsigned int i = 1; i < M; i++) offsets[i] = subs[i-1] + offsets[i-1]; 
        }

        //Precompute all distance between centroids 
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
                // Convert back to UnitVectorFormat and store in codebook
                codebook.push_back(Dataset<UnitVectorFormat>(subspaceSizes[m], dataset.get_size()));
                for (unsigned int i = 0; i < dataset.get_size(); i++) {
                    UnitVectorFormat::Type *c_p = codebook[m][i];
                    float *vec_p = &centroids[i][0];
                    for (unsigned int d = 0; d < subspaceSizes[m]; d++) {
                        *c_p++ = UnitVectorFormat::from_16bit_fixed_point(*vec_p++);
                    }
                }
                
            }
            //showCodebook();
            //showPQCodes();
            //std::cout << "Calculating quantization error for index 1: " << quantizationError(1) << std::endl;
        }


        // Runtime of this is K*dims
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
        
        float asymmetricDistanceComputation(typename UnitVectorFormat::Type* x, typename UnitVectorFormat::Type* y){
            float sum = 0;
            std::vector<uint8_t> px = getPQCode(x);
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
                    for(unsigned int l = 0; l < subspaceSizes[m]; l++){
                        std::cout << "\t" << codebook[m][k][l] << " ";
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