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
- [x] Begin implementation of naive PQ with both euclidean dist optimization and mahalanobis dist. ~ 
  - [x] Euclidean
  - [x] Mahalanobis
- [x] Create Distance table and implement asymmetric and symmetric distance estimator
  - [x] Symmetric
  - [x] Asymmetric
- [ ] Implement random permutation of data points, does it have any effect on (LSH scheme?)
- [x] Working implementation of PQ class to follow format of 'filterer' in index class, and decide design for codebook. i.e. 
  - [x] Compute offsets once 
- [x] Implement simple PQ Code function
- [x] Do not store PQ codes explicitly
- [x] deciding sizes when $d/M \mod 2 \ne 0$ -> we can avoid this issue by carefully selecting m and adding filler 0's on the vectores (this becomes better with random permutation??)
- [x] Begin writing related work for original PQ paper and litterature related to that as well (llyod algo).
- [x] Make PQ and Kmeans work for UnitVectorFormat (and only this as realVectorFormat doesn't work for LSH indexing)
  - [x] Kmeans
  - [x] PQ
- [x] Test quantization error
- [ ] Begin writing formal problem definition of ANN
- [x] Create quick testing setup using acutal data (Investigate if ANN-Benchmark can be used through small datasample and only 1 not all datasets)
- [x] Look at previous bsc. projects of what is included and to what level of expertize.
- [x] Use Float inside kmeans instead of UnitVectorFormat (can be better optimized and shouldn't be a problem for space usage) 
  - [x] Get SIMD  to work for subspaces
  - [x] Use SIMD in mahalanobis distance
- [x] Run experiments and gather data ??
- [ ] Make cost benefit of using SIMD for asymmetric (fraction of padding)
- [ ] Filter passing Criteria for PQFilter (+10exp)
- [ ] Implement PQ as filter with Early stop (martin has to explain code)
- [x] Codebook formatted for faster precomputation
- [x] Make Index work with all similarity measures
- [x] Don't run lloyd multiple times just increase # of iterations
- [ ] Investigate whether distance estimation ever can overflow (might be why we have seen negative values sometimes when inner product was high)


#### Agenda for next meeting
- Explain readings and what we think is important and should be implemented
- Our next steps i.e. TODO list
- Figuring out the threshhold for adding to the buffer?
    - Empirically figure out what works at 'index building time'
    - Bootstrap threshhold such that X\% is above threshhold according to real inner products (maybe faulty as estimated cosine dists are biased??)
    - Other options?




### Agenda for meeting (8/03)
#### Done since last meeting
- Implement mahalanobis
- Use float instead of vectors in kmeans
- Use avx2 instructions where possible (dont fully done - mahalanobis)
- 
#### Current to do
- Make mahalanobis test
- Make explicit where padding is done, how and why.
- Run experiments with both mahalnobis and 
- Gather some results and create simple plots
- Integrate into puffinn for querying
  - boot strapping for treshold values
  - Investigate the usefullness of SIMD when querying (for creating distance table and how lookup is done)
- Perhaps next week should be code review? not sure we have data to discuss yet
- Maybe postpone writing until we are away
- Use SIMD in mahalanobis 

- If to look at more points at first than later lsh rounds, then first figure out if the average similarity becomes smaller in later rounds

#### Issues/Questions
- HPC is 10x slower than my own laptop?
- A lot of code is heavily inspired by NGT, how should I declare this?
- Creating a test for mahalanobis distance is more difficult than we thought. No existing framework does this, and paper has not released source code
- What is the recall plotted in their graphs? actual vs requested (A: actual)

### Conclusion from meeting
- nanoBench for benchmarking see https://github.com/Cecca/puffinn/blob/master/bench/bench.cpp
- Make maha test which performs both euclidean and maha on data and see that whichever is optimized for gives lowest error rate (compare maha error when optimizing for euclidean to when optimizing for maha dist)
- Make simple test that checks that maha distance (isolated) is correct
- Try using cluster again if still slow ask lottie, as it shouldn't be slower
- Remmeber to refer to NGT both in code and in paper
- Maha dist (intuitively) should give lower PQ errors when the data is correlated, and on uncorrelated data it should be exactly the same as euclidean.



 
### Notes from previous Meetings
nanoPQ  
Look more into alignment  
Just find best parameters and optimize for those  
General -> optimize 

### My laptop

#### Comparing -O3 to -O2

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|              114.17 |        8,759,074.33 |    0.2% |      0.00 | `Maha distance WITH AVX -O3`
|              946.36 |        1,056,675.92 |    0.9% |      0.01 | `Maha distance NO AVX -O3`
|              644.36 |        1,551,915.47 |    0.0% |      0.00 | `Maha distance WITH AVX -O2`
|            7,892.78 |          126,698.02 |    0.5% |      0.05 | `Maha distance NO AVX -O2`


#### Comparing Maha to Euc (-O3)
|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|              120.20 |        8,319,147.15 |    0.5% |      0.01 | `Maha distance WITH AVX`
|              902.29 |        1,108,294.78 |    2.8% |      0.05 | `Maha distance NO AVX`
|                4.94 |      202,234,373.16 |    0.1% |      0.00 | `Euc distance WITH AVX`
|               21.43 |       46,653,659.42 |    0.1% |      0.00 | `Euc distance NO AVX`


### HPC

|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
|              813.41 |        1,229,392.79 |    0.2% |        4,967.02 |        2,919.44 |  1.701 |         394.00 |    0.3% |      0.00 | `Maha distance WITH AVX`
|           11,774.51 |           84,929.20 |    0.0% |      107,589.03 |       42,232.22 |  2.548 |      13,581.01 |    0.3% |      0.07 | `Maha distance NO AVX`



Distances stored as floats

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|           14,284.00 |           70,008.40 |    1.7% |      0.00 | `Asymmetric computing PQ code every call`
|               22.72 |       44,022,685.16 |    0.2% |      0.00 | `Asymmetric PQ code precomputed`
|               15.68 |       63,792,633.02 |    0.2% |      0.00 | `Asymmetric fast creating padded query once`
|               30.14 |       33,180,418.53 |    0.3% |      0.00 | `Asymmetric fast creating padded query before each call`
|            6,863.35 |          145,701.44 |    0.7% |      0.00 | `building query distances`
|           17,517.95 |           57,084.30 |    1.9% |      0.00 | `Estimated Inner product O(M)`
|           68,792.95 |           14,536.37 |    1.6% |      0.02 | `True Inner product`

Distances stored as Fixpoints(int16_t)
|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|           15,033.47 |           66,518.26 |    1.1% |      0.36 | `Asymmetric computing PQ code every call`
|               24.70 |       40,491,774.28 |    0.1% |      0.00 | `Asymmetric PQ code precomputed`
|               14.57 |       68,625,813.99 |    1.3% |      0.00 | `Asymmetric fast creating padded query once`
|               29.57 |       33,815,605.24 |    0.4% |      0.00 | `Asymmetric fast creating padded query before each call`
|            6,774.71 |          147,607.79 |    2.9% |      0.16 | `building query distances`
|           16,363.27 |           61,112.49 |    1.7% |      0.39 | `Estimated Inner product O(M)`
|           69,596.97 |           14,368.44 |    0.3% |      1.67 | `True Inner product`
