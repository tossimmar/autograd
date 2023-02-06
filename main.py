import numpy as np
from optim import Adam
from tensor import Tensor
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from module import MLPClassifier

def main():
	'''
	Fit an MLP binary classifier to synthetic concentric cirlces data.
	'''
	
	# two concentric circles centered at the origin with radiuses 1 and 3
	theta = np.linspace(0, 2 * np.pi, 50)
	x1 = np.cos(theta)
	y1 = np.sin(theta)
	x2 = 3 * np.cos(theta)
	y2 = 3 * np.sin(theta)
	x = np.hstack((x1, x2))
	y = np.hstack((y1, y2))
	X = np.vstack((x, y)).T

	# features matrix and target vector
	features = X + np.random.normal(0, 0.15, X.shape)
	target = np.hstack((np.repeat(0, theta.size), np.repeat(1, theta.size)))

	# model, loss function, and optimizer
	model = MLPClassifier(2, 32, 32, 2) # 2D input, two 32D hidden layers, 2D output
	nll_loss = lambda logp, y: -logp[y] # negative log-likelihood
	optimizer = Adam(model.parameters(), lr = 1e-2)

	# training loop
	for epoch in range(201):
		loss = Tensor(np.float64(0))
		for x, y in zip(features, target):
			logp = model(Tensor(x))
			loss = loss + nll_loss(logp, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if epoch % 50 == 0:
		   print(f'Loss {epoch}: {loss}')

	# predictions
	grid = np.array([[x, y] for x in np.linspace(-4, 4, 100) 
		                    for y in np.linspace(-4, 4, 100)])
	pred = [np.argmax(model(Tensor(x)).data) for x in grid]

	# classification boundary plot
	plt.figure(figsize = (5, 5))
	plt.scatter(grid[:, 0], grid[:, 1], c = pred, cmap = 'Pastel1')
	plt.scatter(features[:, 0], features[:, 1], c = target, cmap = 'Pastel1', edgecolor = 'black')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Learned Decision Boundary')
	plt.show()

if __name__ == '__main__':
	main()