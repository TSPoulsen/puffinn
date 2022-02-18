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
        //codebook that contains m*k centroids
        std::vector<Dataset<UnitVectorFormat>> codebook;
        std::vector<std::vector<std::vector<float>>> centroidDistances;
        Dataset<UnitVectorFormat> &dataset;
        std::vector<unsigned int> subspaceSizes, offsets = {0};
        public:
        PQFilter(Dataset<UnitVectorFormat> &dataset, unsigned int dims, unsigned int m = 16, unsigned int k = 256)
        :M(m),
        dims(dims),
        K(k),
        dataset(dataset)
        {
            subspaceSizes.resize(M);
            fill(subspaceSizes.begin(), subspaceSizes.end(), dims/M);
            for(unsigned int i = 1; i < M; i++) offsets.push_back(offsets.back()+ dims/M);
        }
        
        void setSubspaceSizes(std::vector<unsigned int> subs){
            subspaceSizes = subs;
            M = subs.size();
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
                        dists.push_back(UnitVectorFormat::distance(codebook[m][k1], codebook[m][k2], subspaceSizes[m]));
                    }
                    subspaceDists.push_back(dists);
                }
                centroidDistances.push_back(subspaceDists);
            }

        }



        //Runs kmeans for all m subspaces and stores the centroids in codebooks
        void createCodebook(){
            //used to keep track of where subspace begins
            for(unsigned int m = 0; m < M ; m++)
            {
                //RunKmeans for the given subspace
                //gb_labels for this subspace will be the mth index of the PQcodes
                
                KMeans kmeans(dataset, K, offsets[m], subspaceSizes[m]); 
                kmeans.fit();
                codebook.push_back(kmeans.getAllCentroids());
                
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

            return 0.0;
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