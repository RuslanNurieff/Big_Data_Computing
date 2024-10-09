# Big Data Computing projects

The 3 projects are part of my master's class called "Big Data Computing". In the projects, I used Resilient Distributed Dataset (RDD) in order to divide my tasks into several clusters. RDD is the fundamental data structure in PySpark, representing an immutable, distributed collection of objects. It allows data to be processed across a cluster in a fault-tolerant manner. More about RDDs: https://spark.apache.org/docs/latest/rdd-programming-guide.html

## Description

### Project 1

In the first project, I implemented the so-called (M,D)-outlier which is an algorithm for detecting outliers in large datasets. In the code, there are 2 approaches: exact solution and approximate solution. Our outlier definition is as follows:

Let S be a set of N points from some metric space and, for each $p\in S$ let $B_{S}(p,r)$ denote the set of points of S at distance at most r from p. For given parameters M,D>0, an (M,D)-outlier (w.r.t. S) is a point $p\in S$ such that $|B_{S}(p,D)|≤M$. Note that $p\in B_{S}(p,D)$, hence $|B_{S}(p,D)|$ is always at least 1. The problem that we want to study is the following: given S, M, and D, mark each point $p\in S$ as outlier, if it is an (M,D)-outlier, and non-outlier otherwise. For simplicity, we will consider inputs $S\subset R^{2}$ and the standard Euclidean distance.

**Exact Algorithm.** The problem can be solved straighforwardly, by computing all $N(N−1)/2$ pairwise distances among the points, but unfortunately, this strategy is impractical for very large N.

**Approximate Algorithm.** This algorithm is a simple adaptation of an algorithm presented in "Edwin M. Knorr, Raymond T. Ng, V. Tucakov: Distance-Based Outliers: Algorithms and Applications, VLDB J. 8(3-4): 237-253 (2000)." Consider $R^2$ partitioned into square cells of side $\Lambda = D/(2\sqrt{2})$ (whose diagonal length is D/2). For each such cell C, we use an identifier defined by the pair of indices $(i,j)$, with $i,j\in Z$, where $i\cdot \Lambda$ and $j\cdot \Lambda$ are the real coordinates of C's bottom-left corner.  For a point $p=(x_{p},y_{p})\in C$ we define (see also picture below):

- $C_{p}$ = cell where p resides, i.e., $C_{p}=(i,j)$ with $i=\lfloor x_p/\Lambda \rfloor$ and $j=\lfloor y_p/\Lambda \rfloor$.
- $R_{3}(C_{p})$ = 3x3 grid of cells with $C_{p}$ in the middle.
- $R_{7}(C_{p})$ = 7x7 grid of cells with $C_{p}$ in the middle.

![Cell](https://github.com/user-attachments/assets/7c6e5c2e-3968-478b-8440-b1513b6acf49)

Let's also define:

$N3(Cp)$ = number of points in $R3(Cp)∩S$.
$N7(Cp)$ = number of points in $R7(Cp)∩S$.
It is easy to verify that if $N_{3}(C_{p})>M$, then p is a non-outlier, while if $N_{7}(C_{p})≤M$, then p is surely an outlier. Instead, if $N_{3}(C_{p})≤M$ and $N_{7}(C_{p})>M$, then p can be outlier or non-outlier, and we call uncertain. Observe that if an uncertain point is a true outlier, it can be regarded as a "mild" outlier, in the sense that it has more than M points within distance at most 2D.

### Project 2
The program will test a modified version of the approximation strategy for outlier detection developed in Project 1 where the distance parameter D is not provided in input by the user, but is set equal to the radius of a k-center clustering (for a suitable number K of clusters), that is, the maximum distance of a point from its closest center. In other words, the role of the input parameter D is replaced by K. This has two advantages: (1) a better control on the number of non-empty cells; and (2) the potential for a sharper analysis. The k-centers are selected by the Farthers-First-Traversal (FFT) algorithm. The purpose of the project is to assess the effectiveness of this strategy and to test the scalability of a MapReduce implementation when run on large datasets. Originally, the project was run on a cloud machine composed of 16 clusters (CloudVeneto).

### Project 3
For the project, a server was created which generates a continuous stream of integer items. The server is activated on the machine algo.dei.unipd.it and emits the items (viewed as strings) on specific ports (from 8886 or 8889). The program first defines a Spark Streaming Context sc that provides access to the stream through the method socketTextStream which transforms the input stream, coming from the specified machine and port number, into a Discretized Stream (DStream) of batches of items. A batch consists of the items arrived during a time interval whose duration is specified at the creation of the context sc. Each batch is viewed as an RDD of strings, and a set of RDD methods are available to process it. A method foreachRDD is then invoked to process the batches one after the other. Typically, the processing of a batch entails the update of some data structures stored in the driver's local space (i.e., its working memory) which are needed to perform the required analysis. The beginning/end of the stream processing will be set by invoking start/stop methods from the context sc. Typically, the stop command is invoked after the desired number of items is processed.

The program computes:
- The true frequent items with respect to the threshold $\phi$
- An m-sample of $\Sigma$ using Reservoir Sampling of, with $m = \lceil 1/\phi \rceil$
- The epsilon-Approximate Frequent Items computed using Sticky Sampling with confidence parameter $\delta$

More resources about the algorithms can be found under the acknowledgements section.

## Getting Started

### Dependencies

* Python and PySpark should be installed on your machine. Refer to: https://spark.apache.org/docs/latest/api/python/getting_started/install.html

### Installing

* Fork/clone the repository. Each project includes required datasets to test the algorithms (except the 3rd one, at the time of writing, the server may have been shut down).

### Executing program

* Open CMD (on Windows) and for project 1, define the arguments as you wish. The command line takes as arguments path to the data, a float D, and 3 integers M, K, L (in respective order). Here's an example how you can run the program on CMD.
```
python outlier.py data\uber-10k.csv 0.02 10 5 2
```

* For project 2, the procedure is similar (M, K, L) (here L defines the number of clusters we want to use on Cloud):
```
python fft.py data\uber-10k.csv 10 200 16
```

* The procedure for project 3 is a little different since it has no data, but port number for different kind of streams. It receives (through CMD):
  * An integer n: the number of items of the stream to be processed
  * A float phi: the frequency thresold in (0,1)
  * A float epsilon: the accuracy parameter in (0,1)
  * A float delta: the confidence parameter in (0,1)
  * An integer portExp: the port number
```
python sampling.py 1000000 0.07 0.06 0.1 8888
```

## Authors

Ruslan Nuriev

## Acknowledgments

* [Distance Based Outliers](https://www.researchgate.net/publication/225179594_Distance-Based_Outliers_Algorithms_and_Applications)
* [Farthest First Traversal](https://github.com/xuwd11/Coursera-Bioinformatics/blob/master/51_01_FarthestFirstTraversal.py)
* [Reservoir Sampling](https://cesa-bianchi.di.unimi.it/Algo2/Note/reservoir.pdf)
* [Reservoir Sampling & Sticky Sampling](https://www.dei.unipd.it/~geppo/PrAvAlg/DOCS/DFchapter08.pdf)
* [Resilient Distributed Dataset](https://spark.apache.org/docs/latest/rdd-programming-guide.html)
* [Spark Streaming](https://spark.apache.org/docs/latest/streaming-programming-guide.html)
