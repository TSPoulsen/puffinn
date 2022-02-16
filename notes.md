#### Questions
- Which data format should we use? unit\_vector cannot store values larger than 1, which is needed for average vectors and such
- How do we even compile and run the code correctly
- How should we choose M? (Multiple of # of elements in avx2?), or something else with padding inside vectors
- Inertia can overflow? Should we preprocess vectors?
- 
#### Readings
Both [1] does some precomputation by either random permutation or multiplication with an orthonormal matrix, respectively. Why is this?
Seems many approches require linear algebra, shouldn't we use a library? (This would create issues with the currently implemented Dataset storing format)

[1] - Quantization based Fast Inner Product Search



Format for PQ:
- input: Dataset  
- Construct lookup table
- input: query vector
- output: Estimated cosine similarity
- Follow that of the filter already inside puffin that is used to store sketches
- We have to determine threshholds for when data entries are added to the buffer, at least for the sketch distances, not sure how we will handle the stopping criterion

#### TODO:
- [ ] Begin implementation of naive PQ with both euclidean dist optimization and mahalanobis dist. ~ 
- [ ] Create Distance table and implement asymmetric and symmetric distance estimator
- [ ] Implement random permutation of data points, does it have any effect on (LSH scheme?)
- [x] Working implementation of PQ class to follow format of 'filterer' in index class, and decide design for codebook. i.e. 
  - [x] Compute offsets once 
- [x] Implement simple PQ Code function
- [x] Do not store PQ codes explicitly
- [x] deciding sizes when $d/M \mod 2 \ne 0$ (**VIKTOR**) -> we can avoid this issue by carefully selecting m and adding filler 0's on the vectores (this becomes better with random permutation??)
  - [ ] Get SIMD  to work for subspaces
- [ ] Begin writing related work for original PQ paper and litterature related to that as well (llyod algo).
- [ ] Make PQ and Kmeans work for UnitVectorFormat (and only this as realVectorFormat doesn't work for LSH indexing)
- [x] Test quantization error
- [ ] Begin writing formal problem definition of ANN
- [ ] Create quick testing setup using acutal data (Investigate if ANN-Benchmark can be used through small datasample and only 1 not all datasets) (**TIM**)
-[] Look at previous bsc. projects of what is included and to what level of expertize.



#### Agenda for next meeting
- Explain readings and what we think is important and should be implemented
- Our next steps i.e. TODO list
- Figuring out the threshhold for adding to the buffer?
    - Empirically figure out what works at 'index building time'
    - Bootstrap threshhold such that X\% is above threshhold according to real inner products (maybe faulty as estimated cosine dists are biased??)
    - Other options?
- Should PQcodes not be precomputed? It is slower to determine PQcode than it is calculating true inner product




### Agenda for meeting (16/02)
#### Current to do
- built simple test suite to test correctness (*Tim*)
- Move from general TFormat to Unit Vector
- Finish First iteration of PQ (distance between pqcode and q-point)
- Writing broad scope parts of paper (*Viktor*) (not important)

#### what next
  - boot strapping for treshold values
  - get SIMD to work/Optimizing
  - Begin integrating our current solution in the Puffinn pipeline 

#### Questions
- Current Idea for optimizing simd involves adding single point dimensions (In theory it shouldn't mess with correctness?)
- Should we worry about precomputing/exstra-space
- How much should we focus on good code practices vs performance
 
nanoPQ
Look more into alignment
Just find best parameters and optimizae for those
Gerneral -> optimize 
