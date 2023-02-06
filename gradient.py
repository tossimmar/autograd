import utilities
import numpy as np
import scipy.special as ss

# -------------------------------------------------------------------------------------------------

class Neg:
	def __init__(self, output, input):
		self.output = output
		self.input = input

	def backward(self):
		if self.input.grad is None:
			self.input.zero_grad()
		self.input.grad -= self.output.grad

# -------------------------------------------------------------------------------------------------

class Add:
	def __init__(self, output, input1, input2):
		self.output = output
		self.input1 = input1
		self.input2 = input2

	def backward(self):
		if self.input1.requires_grad:
			if self.input1.grad is None:
				self.input1.zero_grad()
			self.input1.grad += self.output.grad
		if self.input2.requires_grad:
			if self.input2.grad is None:
				self.input2.zero_grad()
			self.input2.grad += self.output.grad

# -------------------------------------------------------------------------------------------------

class Sum:
	def __init__(self, output, input):
		self.output = output
		self.input = input

	def backward(self):
		if self.input.grad is None:
			self.input.zero_grad()
		self.input.grad += utilities.inner(np.ones(self.input.size), self.output.grad)

# -------------------------------------------------------------------------------------------------

class Getitem:
	def __init__(self, output, input, slices):
		self.output = output
		self.input = input
		self.slices = slices

	def backward(self):
		if self.input.grad is None:
			self.input.zero_grad()
		slices = self.slices if isinstance(self.slices, tuple) else (self.slices,)
		flat_indices = utilities.slices_to_flat_indices(slices, self.input.shape)
		local_grad = np.zeros((self.input.size, self.output.size))
		for col, row in enumerate(flat_indices):
			local_grad[row, col] = 1
		local_grad = np.squeeze(local_grad)
		self.input.grad += utilities.inner(local_grad, self.output.grad)

# -------------------------------------------------------------------------------------------------

class ReLU:
	def __init__(self, output, input):
		self.output = output
		self.input = input

	def backward(self):
		if self.input.grad is None:
			self.input.zero_grad()
		local_grad = np.diag((self.input.data >= 0).astype(np.float64()).flatten())
		self.input.grad += utilities.inner(local_grad, self.output.grad)

# -------------------------------------------------------------------------------------------------

class LogSoftmax:
	def __init__(self, output, input):
		self.output = output
		self.input = input

	def backward(self):
		if self.input.grad is None:
			self.input.zero_grad()
		local_grad = (np.identity(self.input.size) - ss.softmax(self.input.data).flatten()).T
		self.input.grad += utilities.inner(local_grad, self.output.grad)

# -------------------------------------------------------------------------------------------------

class Linear:
	def __init__(self, output, matrix, vector):
		self.output = output
		self.matrix = matrix
		self.vector = vector

	def backward(self):
		if self.matrix.requires_grad:
			if self.matrix.grad is None:
				self.matrix.zero_grad()
			prefix = self.vector.data
			suffix = np.zeros(self.matrix.size - prefix.size)
			vechot = [np.roll(np.hstack((prefix, suffix)), i * prefix.size) for i in range(self.matrix.shape[0])]
			local_grad = np.vstack(vechot).T
			self.matrix.grad += utilities.inner(local_grad, self.output.grad)
		if self.vector.requires_grad:
			if self.vector.grad is None:
				self.vector.zero_grad()
			self.vector.grad += utilities.inner(self.matrix.data.T, self.output.grad)