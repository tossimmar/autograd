import numpy as np
from tensor import Tensor

# -------------------------------------------------------------------------------------------------

class Affine:
	'''
	Representation of an affine transformation -- that is, 

		x -> Ax + b

	where the matrix A and the vector b are learnable parameters.
	'''
	def __init__(self, in_features, out_features):
		self.weight = Tensor(np.random.normal(0, 1 / np.sqrt(in_features), (out_features, in_features)), requires_grad = True)
		self.bias = Tensor(np.zeros(out_features), requires_grad = True)

	def __call__(self, x):
		return Tensor.linear(self.weight, x) + self.bias

	def parameters(self):
		return [self.weight, self.bias]

# -------------------------------------------------------------------------------------------------

class MLP:
	'''
	Representation of a multilayer perceptron (MLP) with ReLU nonlinearities.
	'''
	def __init__(self, *dims):
		self.affine_layers = [Affine(in_features, out_features) for in_features, out_features in zip(dims[:-1], dims[1:])]
	
	def __call__(self, x):
		for affine in self.affine_layers:
			x = affine(x).relu() if affine is not self.affine_layers[-1] else affine(x)
		return x

	def parameters(self):
		return sum([affine.parameters() for affine in self.affine_layers], [])

# -------------------------------------------------------------------------------------------------
  
class MLPClassifier(MLP):
	'''
	An MLP with log-softmax applied to the output layer.
	'''
	def __call__(self, x):
		return super().__call__(x).log_softmax()