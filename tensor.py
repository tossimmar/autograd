import gradient
import graphlib
import utilities
import numpy as np
import scipy.special as ss

class Tensor:

	_backward_graph = graphlib.TopologicalSorter()

	def _update_backward_graph(grad_fn, output, input1, input2 = None):
		if output.requires_grad:
			output.grad_fn = grad_fn
			Tensor._backward_graph.add(grad_fn) 
			if input1.grad_fn:
				Tensor._backward_graph.add(input1.grad_fn, grad_fn)
			if input2 and input2.grad_fn:
				Tensor._backward_graph.add(input2.grad_fn, grad_fn)

	# ---------------------------------------------------------------------------------------------

	def __init__(self, data, requires_grad = False):
		self.data = data
		self.requires_grad = requires_grad
		self.grad = None
		self.grad_fn = None

	# ---------------------------------------------------------------------------------------------

	def backward(self):
		self.grad = np.float64(1)
		for grad_fn in Tensor._backward_graph.static_order():
			grad_fn.backward()
		Tensor._backward_graph = graphlib.TopologicalSorter()

	# ---------------------------------------------------------------------------------------------

	def zero_grad(self):
		self.grad = np.float64(0)

	# ---------------------------------------------------------------------------------------------

	def __neg__(self):
		output = Tensor(-self.data, requires_grad = self.requires_grad)
		Tensor._update_backward_graph(gradient.Neg(output, self), output, self)
		return output

	# ---------------------------------------------------------------------------------------------

	def __add__(self, other):
		assert(self.shape == other.shape)
		output = Tensor(self.data + other.data, requires_grad = self.requires_grad or other.requires_grad)
		Tensor._update_backward_graph(gradient.Add(output, self, other), output, self, other)
		return output

	# ---------------------------------------------------------------------------------------------

	def __sub__(self, other):
		return self + (-other)

	# ---------------------------------------------------------------------------------------------

	def __getitem__(self, slices):
		if isinstance(slices, tuple):
			assert(self.ndim == len(slices))
		else:
			assert(self.ndim == 1)
		output = Tensor(self.data[slices], requires_grad = self.requires_grad)
		Tensor._update_backward_graph(gradient.Getitem(output, self, slices), output, self)
		return output

	# ---------------------------------------------------------------------------------------------

	def relu(self):
		output = Tensor(np.maximum(self.data, 0), requires_grad = self.requires_grad)
		Tensor._update_backward_graph(gradient.ReLU(output, self), output, self)
		return output

	# ---------------------------------------------------------------------------------------------

	def log_softmax(self):
		output = Tensor(ss.log_softmax(self.data), requires_grad = self.requires_grad)
		Tensor._update_backward_graph(gradient.LogSoftmax(output, self), output, self)
		return output

	# ---------------------------------------------------------------------------------------------

	def sum(self):
		output = Tensor(np.sum(self.data), requires_grad = self.requires_grad)
		Tensor._update_backward_graph(gradient.Sum(output, self), output, self)
		return output

	# ---------------------------------------------------------------------------------------------

	@staticmethod
	def linear(A, x):
		assert(A.ndim == 2 and x.ndim == 1 and A.shape[1] == x.shape[0])
		output = Tensor(A.data @ x.data, requires_grad = A.requires_grad or x.requires_grad)
		Tensor._update_backward_graph(gradient.Linear(output, A, x), output, A, x)
		return output

	# ---------------------------------------------------------------------------------------------

	def __str__(self):
		return str(self.data)

	# ---------------------------------------------------------------------------------------------

	def __getattr__(self, attr):
		return getattr(self.data, attr)