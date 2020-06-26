# Numba how-to
## A collection of useful recipes for Numba

Welcome! Numba is a fantastic library. It compiles Python code to machine code,
 making your functions much faster. However, it cannot compile any Python code.
 Some common Python idioms are not supported yet (but it might be in the 
 future!). Somo common idioms must be adapted in order to achieve high 
 performance.
 
 All recipes are shared as-is, without guarantees. They are merely indicative 
 of what can be done with Numba, and each user must adapt it to their 
 individual use case. These recipes have not undergone extensive testing, 
 and it is up to each user to ensure that they do what they need them to do. 
 
 [Read all the recipes in Read-the-docs](https://numba-how-to.readthedocs.io/en/latest/)
 What
 
 Make tuples from zip, and from list
 
 2d arrays can be 3d arrays with a dimension 1
 
 
