from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import random
import math
import sys

def exact_algorithm(batch, histogram_exact):
    for x_t in batch:
        if x_t in histogram_exact:
            histogram_exact[x_t] += 1
        else:
            histogram_exact[x_t] = 1


def reservoir_sampling(batch, reservoir, t, m):
    for x_t in batch:
        t += 1
        if len(reservoir) < m:
            reservoir.append(x_t)
        else:
            j = random.randint(0, m - 1)
            prob = random.uniform(0, 1)
            if prob <= (m / t):
                reservoir[j] = x_t

def sticky_sampling(batch, sampling_rate, sticky_table):
    for x_t in batch:
        if x_t in sticky_table:
            sticky_table[x_t] += 1
        else:
            prob = random.random()
            if prob < sampling_rate:
                sticky_table[x_t] = 1

def frequent_items_exact(exact_table):
    return {int(item): count for item, count in exact_table.items() if count >= phi * n}

def frequent_items_reservoir(reservoir):
    return [int(item) for item in set(reservoir)]

def frequent_items_sticky_sampling(sticky_table):
    return {int(item): count for item, count in sticky_table.items() if count >= (phi - epsilon) * n}

def process_batch(batch):
    global streamLength, t, histogram_exact, reservoir, sticky_table

    batch_size = batch.count()
    batch_list = batch.collect()
    streamLength += batch_size

    if streamLength > n:
        excess = streamLength - n
        valid_items = batch_size - excess
        
        valid_batch_list = batch_list[:valid_items]
        exact_algorithm(valid_batch_list, histogram_exact)
        reservoir_sampling(valid_batch_list, reservoir, t, m)
        sticky_sampling(valid_batch_list, sampling_rate, sticky_table)
        
        streamLength = n
        
        stopping_condition.set()
    else:
        exact_algorithm(batch_list, histogram_exact)
        reservoir_sampling(batch_list, reservoir, t, m)
        sticky_sampling(batch_list, sampling_rate, sticky_table)

if __name__ == '__main__':
    
    conf = SparkConf().setMaster("local[*]").set("spark.driver.memory", "15g").setAppName("DistinctExample")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event()

    assert len(sys.argv) == 6, "Usage: spark-submit streaming_app.py <n> <phi> <epsilon> <delta> <portExp>"

    # # Read the threshold
    n = sys.argv[1]
    assert n.isdigit(), "n must be an integer"
    n = int(n)

    # # Read the number of clusters.
    phi = float(sys.argv[2])
    assert 0 < phi < 1, "phi must be between 0 and 1"

    epsilon = float(sys.argv[3])
    assert 0 < epsilon < 1, "epsilon must be between 0 and 1"

    delta = float(sys.argv[4])
    assert 0 < delta < 1, "delta must be between 0 and 1"

    portExp = int(sys.argv[5])
    assert isinstance(portExp, int), "Port number must be an integer"

    m = math.ceil(1.0/phi)

    streamLength = 0
    histogram_exact = {}
    reservoir = []
    t = 0
    sticky_table = {}
    r = math.log(1/(delta*phi)) / epsilon
    sampling_rate = r / n

    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda batch: process_batch(batch))  

    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)
    exact_freq_items = sorted(frequent_items_exact(histogram_exact).keys())
    reservoir_freq_items = sorted(frequent_items_reservoir(reservoir))
    sticky_freq_items = sorted(frequent_items_sticky_sampling(sticky_table).keys())
    ########################
    print('INPUT PROPERTIES')
    print("n = " + str(n), "phi = " + str(phi), "epsilon = " + str(epsilon), "delta = " + str(delta), "port = " + str(portExp))
    print("EXACT ALGORITHM")
    print("Number of items in the data structure =", len(histogram_exact))
    print("Number of true frequent items =", len(exact_freq_items))
    print("True frequent items:")
    for i in exact_freq_items:
        print(i)
    ########################
    print("RESERVOIR SAMPLING")
    print("Size m of the sample =", m)
    print("Number of estimated frequent items =", len(reservoir_freq_items))
    print("Estimated frequent items:")
    for i in reservoir_freq_items:
        if i in exact_freq_items:
            print(i, "+")
        else:
            print(i, '-')
    ########################
    print("STICKY SAMPLING")
    print("Number of items in the Hash Table =", len(sticky_table))
    print("Number of estimated frequent items =", len(sticky_freq_items))
    print("Estimated frequent items:")
    for i in sticky_freq_items:
        if i in exact_freq_items:
            print(i, "+")
        else:
            print(i, '-')


