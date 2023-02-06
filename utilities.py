import numpy as np
import itertools as it

# -------------------------------------------------------------------------------------------------

def inner(x, y):
	'''
	Return one of the following products of the specified arguments: 

		1. If either argument is a scalar, return scalar multiplication.
		2. If the first argument is a vector and
			 a. the second argument is a vector, return the inner product.
			 b. the second argument is a matrix, return row vector/matrix multiplication.
		3. If the first argument is a matrix, perform matrix multiplication. 

	arguments
	---------
	x : NumPy array or float
		The first argument of the product.

	y : NumPy array or float
		The second argument of the product.

	returns
	-------
	NumPy array or float
		The product of the specified arguments.
	'''
	return np.inner(x, (y.T if x.ndim and y.ndim == 2 else y))

# -------------------------------------------------------------------------------------------------

def complete_slices(slices, shape):
	'''
	Return complete slices with respect to the specified shape.

	arguments
	---------
	slices : tuple
		Nonnegative integer indices and/or slice objects with respect to the specified shape.

	shape : tuple
		A tuple of dimensions.

	returns
	-------
	tuple
		The complete slices with respect to the specified shape.

	example
	-------
	Let the specified slices and shape be given by

		slices = (1, slice(1, None, None))    and    shape = (3, 3)

	Then, the call complete_slices(slices, shape) returns

		(slice(1, 2, 1), slice(1, 3, 1))
	'''
	assert(len(slices) == len(shape))
	slcs = []
	for s, d in zip(slices, shape):
		if isinstance(s, slice):
			start = s.start if s.start is not None else 0
			stop = s.stop if s.stop is not None else d
			step = s.step if s.step is not None else 1
			slcs.append(slice(start, stop, step))
		else:
			slcs.append(slice(s, s + 1, 1))
	slcs = tuple(slcs)
	return slcs

# -------------------------------------------------------------------------------------------------

def slices_to_flat_indices(slices, shape):
	'''
	Return flat indices with respect to the specified slices and shape.

	arguments
	---------
	slices : tuple
		A tuple of slices with respect to the specified shape.

	shape : tuple
		A tuple of dimensions.

	returns
	-------
	tuple
		The flat indices with respect to the specified slices and shape.
	'''
	assert(len(slices) == len(shape))
	slices = complete_slices(slices, shape)
	indices = it.product(*[range(s.start, s.stop, s.step) for s in slices])
	index_arrays = np.vstack([np.array(index) for index in indices]).T
	flat_indices = tuple(np.ravel_multi_index(index_arrays, shape))
	return flat_indices