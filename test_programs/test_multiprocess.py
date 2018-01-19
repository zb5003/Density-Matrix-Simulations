import multiprocessing as mp
from multiprocessing import Pool
import scipy as sp
import time
import functools

def callback(result, accumulator):
    accumulator += result

def mod_x(y2):
    ret = sp.full((2, 2), y2, dtype=sp.complex128) + sp.full((2, 2), y2 * 1j, dtype=sp.complex128)
    return ret

if __name__ == '__main__':
    t1 = time.time()
    n = 5
    print(mp.cpu_count())
    fuck = sp.zeros((n, 2, 2), dtype=sp.complex128)
    cb = functools.partial(callback, accumulator=fuck)

    with Pool(mp.cpu_count()-1, maxtasksperchild=10) as p:
        for x in range(n):
            p.apply_async(mod_x, (x,), callback=cb)
        p.close()
        p.join()

    print("fuck", fuck, "time = ", time.time() - t1)
