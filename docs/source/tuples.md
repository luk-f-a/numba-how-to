# Working with tuples

## How to zip several tuples

### Problem:
While `zip` works in jitted code, it produces a generator and not a tuple. 
Some functions, like `literal_unroll`, require a tuple to work.

### Solution:

```python
from numba.extending import overload
from numba import types
from numba.extending import intrinsic
from numba.core.cgutils import unpack_tuple

def tuple_zip(*args):
    return tuple(zip(args))

@overload(tuple_zip)
def tuple_zip_ovrl(*args):
    return tuple_zip_intr

@intrinsic
def tuple_zip_intr(tyctx, *tys):
    if len(tys) > 1:
        tys = types.StarArgTuple(tys)
    elif len(tys) == 1:
        raise ValueError("Only one argument received. Tuples to be zipped must be passed as individual arguments")
    nitems = min((x.count for x in tys))
    tuples = [types.Tuple(inner_ty) for inner_ty in zip(*tys)]
    ret = types.Tuple(tuples)
    from numba.core.cgutils import unpack_tuple
    def codegen(cgctx, builder, sig, args):
        assert len(args) == 1  # it is a vararg tuple
        args_tup = unpack_tuple(builder, args[0])
        values = []
        for i in range(nitems):
            inner_vals = [builder.extract_value(x, i) for x in args_tup]
            inner_tup = cgctx.make_tuple(builder, tuples[i], inner_vals)
            values.append(inner_tup)
        return cgctx.make_tuple(builder, sig.return_type, values)
    sig = ret(tys)
    return sig, codegen
```

### Example:

```python

from numba import literal_unroll, njit

@njit
def f1():
    return 1

@njit
def f2():
    return 2

@njit
def f3():
    return 3

@njit
def f4():
    return 4

a = (f1, f2)
b = (f3, f4)
c = (f1, f4)

@njit
def foo(a, b):
    for x in literal_unroll(tuple_zip(a, b, c)):
        f, g, h = x
        print(f()+g()+h())

foo(a, b)

```