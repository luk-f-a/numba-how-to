# Profiling

Profiling jitted code is not yet as convenient as profiling normal python code.
However, here are some ideas about how to use what is currently possible.

## A timer decorator

### Problem
How to measure the time spent in each function.

### Solution
The current solution comes from [Numba's Discourse](https://numba.discourse.group/t/profiling-with-a-decorator-and-njit/55/8).
It uses objmode, which introduces some overhead to each function call due 
to the context switching back into the interpreter.
 This means that it should not be used on very short-running functions, since 
 the overhead will distort the profiling results.  It will also likely have 
 knock-on effects in optimisation which may also skew results. Additionally, 
 it might capture the time spent in compilation, which might or not be 
 relevant in each use case.
 
  
```{code-block} python
import time
import numpy as np
from numba import njit, jit, objmode

results = {}

def jit_timer(f):
    jf = njit(f)
    @njit
    def wrapper(*args):
        with objmode(start='float64'):
            start = time.time()
        g = jf(*args)
        with objmode():
            end = time.time()
            run_time = end - start
            if f.__name__ in results:
                results[f.__name__] += [run_time]
            else:
                results[f.__name__] = [run_time]
        return g
    return wrapper
```

### Example

```{code-block} python
import numpy as np
from numba import njit, jit, objmode

@njit
def pointless_delay(seconds):
    with objmode():
        s = time.time()
        e = 0
        while (e < seconds):
            e = time.time() - s

@jit_timer
def ahp(x, t, u, A):
    pointless_delay(1) # 1s delay
    # total delay is 1s

@jit_timer
def line_intercept(y1, y2, thresh):
    pointless_delay(1) # 1s delay
    # total delay is 1s

@jit_timer
def get_spikes(c, threshold, u, A):
    pointless_delay(2) # 2s delay
    ahp(None, None, None, None) # 1s delay
    line_intercept(None, None, None) # 1s delay
    # total delay is 4s

@jit_timer
def test_sim():
    pointless_delay(7) # 7s delay
    get_spikes(None, 2, 1.2, 100) # 4s delay
    # total delay is 11s

def profile_results():
    l = []
    for k in results:
        a = np.asarray(results[k])
        l += [[k+' '*(13-len(k)), np.sum(a[1:])]]
    l = sorted(l, key=lambda x: x[1])
    for i in range(len(l)):
        print( l[i][0], "{:.6f}".format( l[i][1] ) )

if __name__ == '__main__':
    test_sim()
    test_sim()
    profile_results()
```

## A decorator to track new compilations

### Problem

How to know when new compilations—which can be time consuming—are being performed.

### Solution

The current solution comes [Numba's Discourse](https://numba.discourse.group/t/create-log-message-on-numba-compilation-find-out-if-given-arguments-lead-to-compilation/114/4).
```{code-block} python
from numba import njit
import numpy as np

def logging_jit(func):
    def inner(*args, **kwargs):
        origsigs = set(func.signatures)
        result = func(*args, **kwargs)
        newsigs = set(func.signatures)
        if newsigs != origsigs:
            new = (newsigs ^ origsigs).pop()
             # PUT YOUR LOGGER HERE!
            print("Numba compiled for signature: {}".format(new))
        return result
    return inner

```

### Example

```{code-block} python

@logging_jit
@njit
def foo(a):
    return a + 1

print(foo(4)) # first compile and run for int
print(foo(5)) # second is just a run, int sig is cached
print(foo(6.7)) # third is a compile and run for float
```