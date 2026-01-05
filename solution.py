from collections import Counter
import itertools
from random import randint

def task_func(T1, RANGE=100):
    """
    Convert elements in 'T1' to integers and create a list of random integers where the number of integers 
    is determined by the sum of the integers in `T1`. Random integers are generated between 0 and `RANGE` 
    (default is 100). Count the occurrences of each number in the generated list using a Counter.
    
    Parameters:
    T1 (tuple of tuples): Each inner tuple contains string representations of numbers that are converted to integers.
    RANGE (int, optional): The upper limit for the random number generation. Defaults to 100.
    
    Returns:
    Counter: A Counter object representing the count of each number appearing in the list of generated random integers.
    """
    # Flatten and convert to integers
    ints = [int(x) for x in itertools.chain.from_iterable(T1)]
    total = sum(ints)
    # Generate random numbers
    rand_nums = [randint(0, RANGE) for _ in range(total)]
    return Counter(rand_nums)

