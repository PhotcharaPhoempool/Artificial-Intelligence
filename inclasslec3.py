import numpy as np
import matplotlib.pyplot as plt

mean1 = [-3, -5]
mean2 = [3, 5]
cov1 = np.array([[10, 5], [5, 1]])
cov2 = np.array([[10, 5], [5, 1]])
pts1 = np.random.multivariate_normal(mean1, cov1, size=100)
pts2 = np.random.multivariate_normal(mean2, cov2, size=100)

plt.scatter(pts1[:, 0], pts1[:, 1], marker='.', s=50, alpha=0.5, color='red', label='a')
plt.scatter(pts1[:, 0], pts1[:, 1], marker='.', s=50, alpha=0.5, color='blue', label='b')

plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.grid()
plt.show()