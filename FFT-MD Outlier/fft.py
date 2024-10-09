from pyspark import SparkContext, SparkConf
from math import sqrt, floor
import sys
import os
import numpy as np
import time

conf = SparkConf().setAppName('MRFFT')
conf.set("spark.locality.wait", "0s")
sc = SparkContext(conf=conf)

def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def point_to_cell(point, D):
    size = D / (2 * sqrt(2))
    x, y = point
    id = (floor(x / size), floor(y / size))
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

def MRApproxOutliers(data, D, M):

    point_counts = collect_points(data, D)

    point_counts_dict = point_counts.collectAsMap()
    neighbors_combined = point_counts.map(lambda x: get_neighbor_cells(point_counts_dict, x))

    sure_outliers = (neighbors_combined.filter(lambda x: x[1][2] <= M)
                                      .map(lambda x: x[1][0])
                                      .fold(0, lambda a, b: a + b))


    uncertain_points = (neighbors_combined.filter(lambda x: (x[1][1] <= M and x[1][2] > M))
                                      .map(lambda x: x[1][0])
                                      .fold(0, lambda a, b: a + b))
    
    print("Number of sure outliers =", sure_outliers)
    print("Number of uncertain points =", uncertain_points)

def update_centers(newCenter, data, assignedCenters, dists):
    data = np.array(data)
    newCenter = np.array(newCenter)
    newDists = np.linalg.norm(data - newCenter, axis=1)
    mask = newDists < dists
    assignedCenters[mask] = newCenter
    dists[mask] = newDists[mask]

def SequentialFFT(data, K):
    centers = [data[0]]
    assignedCenters = np.tile(centers[0], (len(data), 1))
    dists = np.linalg.norm(np.array(data) - np.array(centers[0]), axis=1)
    for _ in range(K - 1):
        i = np.argmax(dists)
        newCenter = data[i]
        update_centers(newCenter, data, assignedCenters, dists)
        centers.append(newCenter)
    return centers

def coresets(data, K):
  coresets = (data.mapPartitions(lambda x: SequentialFFT(list(x), K)))
  return coresets.collect()

def distance_to_nearest_center(point, centers):
    min_distance = float('inf')
    for center in centers:
        dist = distance(point, center)
        if dist < min_distance:
            min_distance = dist
    return min_distance

def MRFFT(data, K):
    # Round 1
    time_r1_start = time.time()
    coresets_union_fft = coresets(data, K)
    time_r1_end = time.time()
    time_r1 = 1000 * (time_r1_end - time_r1_start)

    # Round 2
    time_r2_start = time.time()
    coreset_T = SequentialFFT(coresets_union_fft, K)
    time_r2_end = time.time()
    time_r2 = 1000 * (time_r2_end - time_r2_start)

    # Round 3
    time_r3_start = time.time()
    broadcasted_centers = sc.broadcast(coreset_T)
    induced_radius = data.map(lambda x: distance_to_nearest_center(x, broadcasted_centers.value)).reduce(max)
    time_r3_end = time.time()
    time_r3 = 1000 * (time_r3_end - time_r3_start)

    print(f"Running time of MRFFT Round 1 = {time_r1:.0f} ms")
    print(f"Running time of MRFFT Round 2 = {time_r2:.0f} ms")
    print(f"Running time of MRFFT Round 3 = {time_r3:.0f} ms")
    print(f"Radius = {induced_radius:.9f}")

    return induced_radius
    

def main():

    print(sys.argv[1], "M=" + sys.argv[2], "K=" + sys.argv[3], "L=" + sys.argv[4])

    # Read the threshold
    M = sys.argv[2]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    # Read the number of centers
    K = sys.argv[3]
    assert K.isdigit()
    K = int(K)

    # Read the number of partitions
    L = sys.argv[4]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # Read the data
    data_path = sys.argv[1]
    data_rdd = sc.textFile(data_path).map(lambda line: tuple(map(float, line.split(',')))) \
                                     .repartition(numPartitions=L)
    data_count = data_rdd.count()

    print("Number of points =", data_count)

    D = MRFFT(data_rdd, K)
    
    time_approx1 = time.time()
    MRApproxOutliers(data_rdd, D, M)
    print("Running time of MRApproxOutliers = %.0f ms" % (1000 * (time.time() - time_approx1)))


if __name__ == '__main__':
    main()