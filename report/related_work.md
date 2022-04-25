## Related Work
ANN algorithms often consists of many parts; an indexing algorithm to find good candidates for nearest neighbours such that not all points have to be searched, and methods for estimating similarity between candidate points.
The latter group of similarity estimation methods is known as sketching. It provides a way of quickly estimating similarities.
Calculating the true distance between points is $\Theta (d)$ where $d$ is the number of dimensions of a data entry. 
When $d$ is large calucating true distances becomes computationally heavy as this has to be done for all candidates that an indexing algorithm suggests.
In the following sections we will outline some common skething techniques used today, and thereafter explain in depth the technique used in this paper.

We will focus solely on cosine similarity as our similiarity measure, but when all points are normalized to lie on the unit sphere, both cosine similarity and inner product sketching techniques can be used, as the two similarity measures become equivalent.

### Skething techniques

#### Locality Sensitive Hashing

Locality sensitive hashing (LSH) methods have seen much success in the setting of ANN over the last decade.  
Put simply:  
<br>
*"A locality-sensitive hash (LSH) family $\mathcal{H}$ is family of functions $h : X \rightarrow R$,
such that for each pair $x, y \in X$ and a random $h \in H$, for arbitrary $q \in X$,
whenever $dist(q, x) \leq dist(q, y)$ we have $p(q, x) := Pr [h(q) = h(x)] \geq Pr [h(q) = h(y)]$"* &nbsp; [1]

This means that the more similiar points are the more likely their hash value is to be the same (Rephrase sentence).
The LSH-family currently used in PUFFINN for sketching is SimHash.  
SimHash was originally introduced in [2] (I think) and the variant used in PUFFINN works by creating a random hyperplane at the origin, and then creates a 1-bit hash depending on which side of the hyperplane a point is located.
This is then done $w$ times to create sketches of size $w$.
Similarity is then estimated by counting the number of collisions between two such sketches. The more collisions the higher the angular similarity. As these 1-bit hashes can be packed into 64-bit integers, estimating the similarity can be performed quickly using cpu instructions for hamming distance bewteen binary values.


#### Concomitants 

The paper [3] introduces a new method that uses concomitants of extreme order statistics.
Let $r,x,q \in \mathbb{R}^d$ where $r$ is a Guassian random vector with coordinates sampled from $N(0,1)$, $q$ is a query vector and $x$ is a database vector.
Performing $Q = q^Tr$ and $X = x^Tr$  against $D$ instances of $r$ creates a set of pairs,
in where the concomitant of the highest value of $Q$,(I don't get this)
referred to as $X_{[1]}$, has the property that $\mathbb{E} \left[ \frac{X_{[1]}}{\sqrt{2logD}} \right] = q^Tx$. Thus meaning it can be used as an unbiased estimator for the inner product between two vectors, regardless of distributions of queries and database.


### Product Quantization

<br>

All the product quantization techniques we will discuss perform _k_-means clustering, and the most well known procedure for performing _k_-means clustering is the Llyod algorithm
The algorithm in all its simplicity can be described as
- __Intitialize centroids__; either through random sampling points or using the kmeans++ initialization method  
- __Partition points__ such that each point is partitioned to the centroid that minimizes the chosen loss function $x_{k}=arg\min_{k}\mathcal{d}(x,c_{k})$
- __Update centroids__ such that minimal average loss is achieved between the centroid and all points partitioned to it.

The last 2 steps are iterated until either convergance or some stopping criterion is reached.  
The main differences between the following methods, are their choice of loss function for the llyod algorithm.

#### Origin of Product Quantization


Produc Quantization (PQ) has been around for many years, and a general outline and history of it can be found in [6].
PQ produce sketches by dividing each vector into $M$ subspaces, most often a point's $M$ cartesian subspaces.
Then each subspace is then quantized independently into $K$ centroids (other word than centroids?), thus forming the codebook $C$.
Each subspace is most often quantized using the Llyod algorithm, to produce the $K$ centroids.  
The indicies of each centroid that a point is closest to in each subspace, make up the sketch of that point.   
More formally the sketch of a point is: (IT IS NOT A SET, BUT I DON'T KNOW HOW ELSE TO WRITE IT)
$$ \hat{x} = \{ c_x^{(0)}, c_x^{(1)}, ..., c_x^{(M)}\}$$
Where $c^{(m)}_x$ is the index of the centroid that data point $x$ is partitioned to in subspace $m$, thus $c^{(m)}_x = arg\min_k\mathcal{d}(x^{(m)},C^{(m)}_k)$, where $C^{(m)}_k$ is the $k$ 'th centriod in subspace $m$ and $\mathcal{d}$ is the distance function.

The bit-size of a sketch is therefore determined by $\lceil \log_2 K \rceil \cdot  M$. Common values of $K$ and $M$ are $256$ and $8$ respectively, which means sketches are 64-bit.
Reconstruction of a point is then done by concatenating the $M$ centriods gotten from the sketch into a single vector 
$$\hat{x} = \left[ C[c^{(0)}_x];C[c^{(1)}_x];...;C[c^{(M)}_x]  \right]$$
and the reconstruction error is the defined as: $e_x = \ell(x, \hat{x})$ Which is the loss function that in the end is used to derive the distance function that minimizes total loss in the Llyod algorithm.

In the context of ANN, PQ works well because the distance bewteen a query and all centroids can be precomputed in $O(D \cdot K)$ and then distance between any given point and the query can be calculated in $O(M)$.
So for a query set of size $Q$ and a dataset of size $N$ and each point has dimension $D$, distances between all queries and all points in the dataset can be done in $O(Q  D  K + N  M)$ which is compared to $O(Q  D  N)$ for computing true distances.
When $N \gg K$ using PQ as a sketching technique is much less computationally expensive.

[4] was the first paper to use Product (or also Called vector-?) quantization (PQ) for nearest neighbour search.
In their paper they use Mean Squared Error (MSE) as their loss function, which leads the distance function to be euclidean distance.
Their results show that PQ gives good estimates for euclidean distance, and proved both efficient and effective. 
They compare two ways of computing distances between a query and a quantized dataset, called Asymmetric- vs Symmetric distance computation (ADC vs SDC).
The two methods differ in that SDC quantize the query vector using the codebook learned from the dataset, and then calculates the distance to points using the quantized version of the query, whereas ADC calculates the distances using the original query.

Both their theoretical and emperical results find that using the asymmetric distance was better, as it lead to a lesser biased distance estimator and also higher recalls when using it for ANN search.
It discusses the theoretical time and memory complexities for both SDC, ADC and show that they are equal in both encoding a vector and estimating distance to a query.

(There is a whole section on using a correction term for ADC, but they find it is unneccesary)



#### PQ for inner product search 


[5] shows that if one defines the loss function as the squared error between true inner product and estimated inner product:

$$ \ell(x,q) = \left(q^Tx - \sum_{m}{q^{(m)T}C[c_x^{(m)}]} \right)^2 $$

The two iterated steps in the llyod algorithm are then performed by partitioning points according to:

$$ c_x^{(m)} = \arg\min_k \left( x^{(m)} - C^{(m)}_k \right) \Sigma_X^{(m)} \left( x^{(m)} - C^{(m)}_k \right) $$

where $\Sigma_X^{(k)}$ is the non-centered covariance matrix in subspace $m$  
and updating centroids according to:

$$ C^{(m)}_k = \frac{1}{|S_k^{(m)}|} \sum_{x^{(m)} \in S_k^{(m)}} x^{(m)} $$

They also prove that the update step leads $\sum_{m}{q^{(m)T}C[c_x^{(m)}]}$ to be an unbiased estimator of $q^Tx$.
This all relies upon the assumptions that the cartesian subspaces are independent and that the queries come from the same distribution as that of database vectors.

## Extra
They show that having the codebook consist of the euclidean means of vectors in each clsuters then using the codebook for inner product estimation is an unbiased estimator (lemma 3.1).
This means that 
$$ \mathbb{E} \left[ \sum_{m}{q^{(m)T}u_x^{(m)}} \right] = q^Tx$$
holds if 
$$ C^{(m)}_k = \frac{1}{|S_k^{(m)}|} \sum_{x^{(m)} \in S_k^{(m)}} x^{(m)} $$
where $S_k^{(m)}$ is the set of points partinioned to centroid $C^{(m)}_k$

This fact thus leads to the realization that in the classical llyod algorithm for k-means clustering (and therefore codebook learning), it is how points are assigned that influences the accuracy of the codebook, if an unbiased estimator is desired.

The last two are what satisfy Llyod optimality conditions

Formalities:
- $K$ is number of centroids $k$ is a specific centroid
- $M$ is number of subspaces $m$ is a specific subspace (cartesian)
- $C$ is the whole Codebook and $C^{(m)}$ is the codebook for subspace $m$ and $C^{(m)}_k$ is the _k_'th codebook centroid in the _m_'th subspace
- $c^{(m)}_x$ is the index of the centroid that data point $x$ is partitioned to in subspace $m$, thus $c^{(m)}_x = arg\min_k\ell(x^{(m)},C^{(m)}_k)$


for_editor


[1] PUFFINN: Parameterless and Universally Fast FInding of Nearest Neighbors  
[2] Similarity estimation techniques from rounding algorithms  
[3] Sublinear Maximum Inner Product Search using Concomitants of Extreme Order Statistics (Ninh Pham)  
[4] Product quantization for nearest neighbor search (Schmid et al)  
[5] Quantization based Fast Inner Product Search (Guo et al)  
[6] Quantization (David L. Neuhoff et al)

