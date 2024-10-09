from pyspark import SparkContext, SparkConf
from math import sqrt
import sys
import os
import numpy as np
import timeit

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def display_points(outliers, K):
    print("Number of Outliers:", len(outliers))
    for idx, (point, _) in enumerate(outliers):
        if idx >= K:
            break
        print(f"Point: {point}")

def ExactOutliers(points, D, M, K):
    D = D ** 2
    outliers = list()
    neighbours = np.ones(len(points), dtype=int)
    for i, point in enumerate(points):
        for j in range(i+1, len(points)):
            dist = distance(point, points[j])
            if dist <= D:
                neighbours[j] += 1
                neighbours[i] += 1
        
    for t in range(len(points)):
        if neighbours[t] <= M:
            outliers.append((points[t], neighbours[t]))

    sorted_outliers = sorted(outliers, key=lambda x: (x[1], x[0]))

    display_points(sorted_outliers, K)

def point_to_cell(point, D):
    size = D / (2 * sqrt(2))
    x, y = point
    id = (int(x / size), int(y / size))
    return (id, 1)

def collect_points(points_rdd, D):
    points = (points_rdd.map(lambda x: point_to_cell(x, D))
                        .reduceByKey(lambda x, y: x + y))
    return points

def get_neighbor_cells(points_dict, point):
    cell_id, cell_count = point
    i, j = cell_id
    neighbors_count_3x3 = sum([points_dict.get((i + ni, j + nj), 0) for ni in range(-1, 2) for nj in range(-1, 2)])
    neighbors_count_7x7 = sum([points_dict.get((i + ni, j + nj), 0) for ni in range(-3, 4) for nj in range(-3, 4)])
    return (cell_id, (cell_count, neighbors_count_3x3, neighbors_count_7x7))

def MRApproxOutliers(data, D, M, K):

    point_counts = collect_points(data, D)

    point_counts_dict = point_counts.collectAsMap()
    neighbors_combined = point_counts.map(lambda x: get_neighbor_cells(point_counts_dict, x))

    sure_outliers = (neighbors_combined.filter(lambda x: x[1][2] <= M)
                                      .map(lambda x: x[1][0])
                                      .reduce(lambda a, b: a + b))


    uncertain_points = (neighbors_combined.filter(lambda x: (x[1][1] <= M and x[1][2] > M))
                                      .map(lambda x: x[1][0])
                                      .reduce(lambda a, b: a + b))
    
    print("Number of sure outliers = ", sure_outliers)
    print("Number of uncertain points = ", uncertain_points)

    if K > point_counts.count():
        K = point_counts.count()
        sorted_cells = point_counts.map(lambda x: (x[1], x[0])).sortByKey().take(K)
    else:
        sorted_cells = point_counts.map(lambda x: (x[1], x[0])).sortByKey().take(K)

    for cell in sorted_cells:
        print(f"Cell: {cell[1]} Size: {cell[0]}")
    

def main():
    # SPARK SETUP
    conf = SparkConf().setAppName('WordCountExample')
    sc = SparkContext(conf=conf)

    assert len(sys.argv) == 6, "Usage: python outlier_detection.py <K> <file_name>"

    print(sys.argv[1], "D = " + sys.argv[2], "M = " + sys.argv[3], "K = " + sys.argv[4], "L = " + sys.argv[5])

    # Read the distance
    D = sys.argv[2]
    D = float(D)

    # Read the threshold
    M = sys.argv[3]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    # Read the number of points that need to be shown
    K = sys.argv[4]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # Read the number of partitions
    L = sys.argv[5]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # Read the data
    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    data_rdd = sc.textFile(data_path).map(lambda line: tuple(map(float, line.split(',')))) \
                                     .repartition(numPartitions=L)
    data_count = data_rdd.count()

    print("Number of points = ", data_count)
    
    if data_count <= 200000:
        with open(data_path, 'r') as f:
            listOfPoints = [tuple((map(float, i.split(',')))) for i in f]
        time_exact1 = timeit.default_timer()
        ExactOutliers(listOfPoints, D, M, K)
        print("Running time of ExactOutliers = %.1f ms" % (1000 * (timeit.default_timer() - time_exact1)))

        time_approx1 = timeit.default_timer()
        MRApproxOutliers(data_rdd, D, M, K)
        print("Running time of MRApproxOutliers = %.1f ms" % (1000 * (timeit.default_timer() - time_approx1)))
    else:
        time_approx1 = timeit.default_timer()
        MRApproxOutliers(data_rdd, D, M, K)
        print("Running time of MRApproxOutliers = %.1f ms" % (1000 * (timeit.default_timer() - time_approx1)))


if __name__ == '__main__':
    main()