


# Recreating results and plots
#### Prerequisites
- HDF5-serial
    - Can be installed with `sudo apt install libhdf5-dev`
- cmake 3.6 or greater

Getting the data can be done through:
```
mkdir data/
cd data
wget http://ann-benchmarks.com/glove-100-angular.hdf5
wget http://ann-benchmarks.com/nytimes-256-angular.hdf5
wget http://ann-benchmarks.com/deep-image-96-angular.hdf5
cd ..
```


To create the plots use python=3.9 (others should work as well) and
install dependencies by `pip install -r requirements.txt`

To create the estimates using our PQ Filter along with their true inner product
have your current working dir be the root of the project and run the following:
```
cmake .
make Quick
./Quick data/<dataset> <M> <loss> <permutation>
```
and example of such a command is
`./Quick data/glove-100-angular 8 euclidean perm`  
options for `<loss>` are `euclidean` and `mahalanobis`  
options for `<permutation>` are `perm` whereas everyting else will lead to no permutation being done   
The results will be written to the file `experiments/results/<dataset>_<loss>_<M>_<perm>.hdf5`

Recreating the same for the HP-LSH filter create the executable `Quick` in the same manner and execute  
`./Quick data/<dataset> <N_sketches>`.  
`N_sketches` specify how many sketches to use, where each is 64 bits. This produces the following file `experiments/results/<dataset>_lsh_<N_sketches>.hdf5`.
This contains the true inner products between queries and all points in the dataset, along with the collisions between the `N_sketches` of both query and data point.  
NOTE: This is not normalized and needs to be done in the python code to get the collision probability.





Create plots can be done by going through the notebook `plots.ipynb`


