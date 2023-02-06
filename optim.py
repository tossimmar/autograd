import numpy as np

class Adam:
	'''
	Adam optimization algorithm.
	
	See: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
	'''
	def __init__(self, parameters, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, weight_decay = 0):
		self.parameters = {parameter: {'t': 1, 'mt': 0, 'vt': 0} for parameter in parameters}
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.weight_decay = weight_decay
	
	def step(self):
		for parameter, d in self.parameters.items():
			gt = np.reshape(parameter.grad, parameter.shape)
			if self.weight_decay:
				gt += self.weight_decay * parameter.data
			d['mt'] = self.beta1 * d['mt'] + (1 - self.beta1) * gt
			d['vt'] = self.beta2 * d['vt'] + (1 - self.beta2) * gt ** 2
			mt_hat = d['mt'] / (1 - self.beta1 ** d['t'])
			vt_hat = d['vt'] / (1 - self.beta2 ** d['t'])
			parameter.data -= self.lr * mt_hat / (np.sqrt(vt_hat) + 1e-8)
			d['t'] += 1
	
	def zero_grad(self):
		for parameter in self.parameters:
			parameter.zero_grad()
