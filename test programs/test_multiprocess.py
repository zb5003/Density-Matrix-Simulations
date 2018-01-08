import multiprocessing as mp
import scipy as sp
import time

x = 5
rando = sp.full(5, 1 + 1j, dtype=complex)

def print_fuck(y):
    print("fuck", y)
    return None

def mod_x(y2, a_re, a_im):
    y1 = x + y2
    a_re[:] = [i + j for i, j in zip(a_re[:], rando.real)]
    a_im[:] = [i + j for i, j in zip(a_im[:], rando.imag)]
    # for i in range(5):
    #     arr[i] = arr[i] + 3
    b = 0
    for i in range(100000000):
        y1 = y1 + 1
    return None

print(__name__)
if __name__ == '__main__':
    a_real = mp.Array('d', 5, lock=False)
    a_imag = mp.Array('d', 5, lock=False)

    jobs = []
    t1 = time.time()
    for i in range(5):
        p = mp.Process(target=mod_x, args=(2, a_real, a_imag))
        # jobs.append(p)
        p.start()
    p.join()
    print(a_real[:], a_imag[:])
    a = sp.asarray([i + 1j * j for i, j in zip(a_real[:], a_imag[:])], dtype=complex)
    print("fuck", a[:], "time = ", time.time() - t1)
