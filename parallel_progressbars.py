'''
Multiple progress bars for monitoring parallel jobs

'''


import time
import random
from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock
from joblib import Parallel, delayed
import multiprocessing

N = 10000
def func(pid, param):

    tqdm_text = "#" + "{}".format(param).zfill(1)

    current_sum = 0
    with tqdm(total=N, desc=tqdm_text, position=pid+1) as pbar:
        for i in range(1, N+1):
            current_sum += i
            time.sleep(0.001)
            pbar.update(1)
    
    return current_sum

def main():

    inputs = [10, 20, 30, 40]
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(func)(i, param) for i, param in enumerate(inputs))

    

if __name__ == "__main__":

    main()