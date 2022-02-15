#pragma once

#include "puffinn/dataset.hpp"
#include "puffinn/kmeans.hpp"
#include <vector>
#include <cfloat>
#include <iostream>
namespace puffinn{
    template<typename TFormat>
    class PQFilter{
        const unsigned int M, dims;
        const unsigned char K;
        //codebook that contains m*k centroids
        std::vector<Dataset<TFormat>> codebook;
        Dataset<TFormat> &dataset;
        std::vector<std::vector<uint8_t>> pqCodes;
        std::vector<unsigned int> subspaceSizes;
        public:
        PQFilter(Dataset<TFormat> &dataset, unsigned int dims, unsigned int m = 16, unsigned int k = 256)
        :dataset(dataset),
        dims(dims),
        M(m),
        K(k),
        pqCodes(dataset.get_size())
        {
            subspaceSizes.resize(M);
            fill(subspaceSizes.begin(), subspaceSizes.end(), dims/M);
        }
        
        void setSubspaceSizes(std::vector<unsigned int> subs){
            subspaceSizes = subs;
        }


        //Runs kmeans for all m subspaces and stores the centroids in codebooks
        void createCodebook(){
            //used to keep track of where subspace begins
            int offset = 0;
            for(unsigned int subSize: subspaceSizes)
            {
                //RunKmeans for the given subspace
                //gb_labels for this subspace will be the mth index of the PQcodes
                
                KMeans<TFormat> kmeans(dataset, K, offset, subSize); 
                kmeans.fit();
                codebook.push_back(kmeans.getAllCentroids());
                uint8_t * labels = kmeans.getLabels();
                for(int i = 0; i < dataset.get_size(); i++){
                    pqCodes[i].push_back(labels[i]);
                }
                offset += subSize;
            }
            //showCodebook();
            //showPQCodes();
            //std::cerr << "Calculating quantization error for index 1: " << quantizationError(1) << std::endl;
        }


        std::vector<uint8_t> getPQCode(int index){
            return pqCodes[index];
        }

        float quantizationError(int index){
            float sum = 0;
            typename TFormat::Type* vec = dataset[index];
            int centroidID, offset = 0;

            for(int m = 0; m < M; m++){
                centroidID = pqCodes[index][m];
                sum += TFormat::distance(vec + offset, codebook[m][centroidID], subspaceSizes[m]);
                offset += subspaceSizes[m];
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
            for(int i  = 0; i < dataset.get_size(); i++){
                sum += quantizationError(i);
            }
            return sum;
        }

        std::vector<float> getCluser(unsigned int mID, unsigned int kID){
            return std::vector<float>(codebook[mID][kID], codebook[mID][kID]+ subspaceSizes[mID]);
        }

        //Functions below are just debugging tools 
        void showCodebook(){
            for(int m = 0; m < M; m++){
                std::cerr << "subspace: " << m << std::endl;
                for(int k = 0; k < K; k++){
                    std::cerr << "cluster: "<< k << std::endl;
                    for(int l = 0; l < subspaceSizes[m]; l++){
                        std::cerr << "\t" <<codebook[m][k][l] << " ";
                    }
                    std::cerr << std::endl;
                }
                std::cerr << std::endl;
            } 
        }

        void showPQCodes(){
            std::cerr << "PQCODE: ";
            for(std::vector<uint8_t> pqCode: pqCodes){
                for(uint8_t val: pqCode){
                    std::cerr << (unsigned int) val << " ";
                }
                std::cerr << std::endl;
            }

        }

        void showSubSizes(){
            for(auto a: subspaceSizes){
                std::cerr << a << " ";
            }
            std::cerr << std::endl;
        }
    };
}