# Working with Numpy

Numba is very well integrated with Numpy and supports a wide range of Numpy 
functions. However, some functions are still unsupported.

## Using the new (1.17+) numpy.random Generator

### Problem

Numba cannot use the new `random` subpackage or its `Generator` objects.

```{code-block} python
import numpy as np
from numba import njit

@njit
def foo():
    rng = np.random.default_rng()
    return rng.standard_normal(10)

foo()

---------------------------------------------------------------------------
TypingError                               Traceback (most recent call last)


TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Unknown attribute 'default_rng' of type Module(<module 'numpy.random')
```

### Solution

Use the `objmode` context manager ([docs here](https://numba.pydata.org/numba-doc/latest/user/withobjmode.html)).

### Example
```{code-block} python
import numpy as np
from numba import njit, objmode

@njit
def foo():
    with objmode(y="float64[:]"):
        rng = np.random.default_rng()
        y=rng.standard_normal(10)
    return y

foo()

---------------------------------------------------------------------------
TypingError                               Traceback (most recent call last)


TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Unknown attribute 'default_rng' of type Module(<module 'numpy.random')
```

Even though the context manager introduces a small overhead, for large arrays 
the time that it takes to generate the random number will completely dominate.
 