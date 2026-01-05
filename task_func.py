import numpy as np
import itertools
import random
import statistics

def task_func(T1, RANGE=100):
    """
    Convert elements in 'T1' to integers and create a list of random integers.
    The size of the list is the sum of the integers in `T1`. Calculate and 
    return the mean, median, and mode of the list.
    
    Parameters:
    T1 (tuple of tuples): Each tuple contains string representations of integers which are converted to integers.
    RANGE (int, optional): The upper limit for generating random integers. Default is 100.
    
    Returns:
    tuple: A tuple containing the mean, median, and mode of the generated list of random integers.
           The mean and median are floats, and the mode is an integer. The calculations use the generated
           list whose size is determined by the sum of converted integers from `T1`.
    
    Raises:
    statistics.StatisticsError if T1 is empty
    """
    if not T1:
        raise statistics.StatisticsError('T1 is empty')
    # Flatten and convert to integers
    flat_iter = itertools.chain.from_iterable(T1)
    ints = [int(x) for x in flat_iter]
    total = sum(ints)
    # Generate random list of specified size
    rand_list = [random.randint(0, RANGE) for _ in range(total)]
    # Compute statistics
    mean_val = statistics.mean(rand_list)
    median_val = statistics.median(rand_list)
    mode_val = statistics.mode(rand_list)
    return (float(mean_val), float(median_val), int(mode_val))

