# Working with tuples

## How to zip several tuples

### Problem:
While `zip` works in jitted code, it produces a generator and not a tuple. 
Some functions, like `literal_unroll`, require a tuple to work.

### Solution:

```{code-block} python
from numba.extending import overload
from numba import types
from numba.extending import intrinsic
from numba.core.cgutils import unpack_tuple

def tuple_zip(*args):
    return tuple(zip(args))

@overload(tuple_zip)
def tuple_zip_ovrl(*args):
    return lambda *args: tuple_zip_intr(*args)

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

```{code-block} python

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

## How to use tuples of different length as keys in a dictionary

### Problem: 
The keys of a dictionary must be all of the same type. The type of a tuple is determined by its length,
and therefore tuples of different length cannot be used as keys in the same dictionary.

### Solution:

```{code-block} python
from numba import njit, literal_unroll, types
from numba.typed import Dict
import numpy as np
from numba.experimental import structref
from numba.extending import overload
import operator

# The idea here is to wrap a typed.Dict in another type, the "TupleKeyDictType".
# The purpose of this is so that operations like __getitem__ and __setitem__
# can be proxied through functions that call `hash` on the key. This makes it
# possible to have something that behaves like a dictionary, but supports
# heterogeneous keys (tuples of varying size/type).

# Define a the new type and register it
@structref.register
class TupleKeyDictType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

# Define the Python side proxy class
class TupleKeyDict(structref.StructRefProxy):
    @property
    def wrapped_dict(self):
        return TupleKeyDict_get_wrapped_dict(self)

# Set up the wiring for it, "wrapped_dict" is the only member in the "struct"
# and it refers to the typed.Dict instance in use
structref.define_proxy(TupleKeyDict, TupleKeyDictType, ["wrapped_dict"])

# Overload operator.getitem for the TupleKeyDictType, note how defers the look
# up to the wrapped_dict member and hashes the key
@overload(operator.getitem)
def ol_tkd_getitem(inst, key):
    if isinstance(inst, TupleKeyDictType):
        def impl(inst, key):
            return inst.wrapped_dict[hash(key)]
        return impl

# Overload operator.setitem for the TupleKeyDictType, again, it's hashing the
# key before use.
@overload(operator.setitem)
def ol_tkd_setitem(inst, key, value):
    if isinstance(inst, TupleKeyDictType):
        def impl(inst, key, value):
            inst.wrapped_dict[hash(key)] = value
        return impl


```

### Example:

```{code-block} python
@njit
def foo(keys, values):
    # Create a dictionary to wrap
    wrapped_dictionary = Dict.empty(types.intp, types.complex128)
    # wrap it
    tkd_inst = TupleKeyDict(wrapped_dictionary)

    # Add some items, this is a bit contrived...
    # keys is heterogeneous in dtype (different sized tuples) so needs loop
    # body versioning for iteration (i.e. literal_unroll).
    idx = 0
    for k in literal_unroll(keys):
        tkd_inst[k] = values[idx]
        idx += 1

    # print the wrapped instance
    print(tkd_inst.wrapped_dict)

    # demo getitem
    print("getitem", (1, 2), "gives", tkd_inst[(1, 2)])

keyz = ((1, 2), (3, 4, 5), (6,))
valuez = (1j, 2j, 3j)

foo(keyz, valuez)
```
