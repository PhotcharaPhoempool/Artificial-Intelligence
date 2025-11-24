import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# PART 1 : Classwork หน้า 18
#   เปรียบเทียบการใช้ 1 feature กับ 100 feature ด้วย Bayes Classifier
# ============================================================

n_per_class = 250       # วัตถุชนิดละ 250 ชิ้น
n_features  = 100       # มี feature ทั้งหมด 100 ชนิด

# กำหนด mean / variance ของแต่ละ class (ทุก feature)
mu_A  = np.zeros(n_features)
mu_B  = np.zeros(n_features)
var_A = np.full(n_features, 0.75)   # feature ที่ไม่ช่วยแยก: N(0, 0.75)
var_B = np.full(n_features, 0.75)

# feature ที่ 1 เป็นตัวที่ใช้แยก class
# Class A ~ N(3, 0.10) , Class B ~ N(6, 0.75)
mu_A[0]  = 3.0
mu_B[0]  = 6.0
var_A[0] = 0.10
var_B[0] = 0.75

# สุ่มข้อมูลจาก Gaussian (feature แต่ละตัว independent)
X_A = np.random.normal(mu_A, np.sqrt(var_A), size=(n_per_class, n_features))
X_B = np.random.normal(mu_B, np.sqrt(var_B), size=(n_per_class, n_features))

X = np.vstack([X_A, X_B])
y = np.array([0]*n_per_class + [1]*n_per_class)   # 0 = Class A, 1 = Class B

def log_gaussian_prob(x, mu, var):
    return -0.5 * (np.log(2*np.pi*var) + (x - mu)**2 / var)

def bayes_predict_highdim(X_in, use_all_features=True):
    feats = slice(None) if use_all_features else slice(0, 1)
    x   = X_in[:, feats]
    muA = mu_A[feats]
    muB = mu_B[feats]
    vA  = var_A[feats]
    vB  = var_B[feats]

    log_pA = np.sum(log_gaussian_prob(x, muA, vA), axis=1) + np.log(0.5)
    log_pB = np.sum(log_gaussian_prob(x, muB, vB), axis=1) + np.log(0.5)
    return (log_pB > log_pA).astype(int)

acc_1   = np.mean(bayes_predict_highdim(X, use_all_features=False) == y)
acc_100 = np.mean(bayes_predict_highdim(X, use_all_features=True)  == y)

print('=== Classwork p.18 ===')
print(f'Accuracy ด้วย 1 feature   = {acc_1:.3f}')
print(f'Accuracy ด้วย 100 feature = {acc_100:.3f}\n')

# ============================================================
# PART 2 : Classwork หน้า 19–32
#   Bayes classification for 2D multivariate normal data
# ============================================================

# Data Representation: 2D Gaussian 2 classes
mean0 = np.array([0.0, 0.0])      # Class 0
mean1 = np.array([3.2, 0.0])      # Class 1

Sigma0 = np.array([[0.10, 0.0],
                   [0.0,  0.75]])  # Class 0 covariance
Sigma1 = np.array([[0.75, 0.0],
                   [0.0,  0.10]])  # Class 1 covariance

N0 = 200
N1 = 200

X0 = np.random.multivariate_normal(mean0, Sigma0, size=N0)
X1 = np.random.multivariate_normal(mean1, Sigma1, size=N1)

# Estimate parameters
mu0_hat = X0.mean(axis=0)
mu1_hat = X1.mean(axis=0)
Sigma0_hat = np.cov(X0, rowvar=False)
Sigma1_hat = np.cov(X1, rowvar=False)

print('=== Estimated parameters (p.20–21) ===')
print('mu0_hat =', mu0_hat)
print('mu1_hat =', mu1_hat)
print('Sigma0_hat =\n', Sigma0_hat)
print('Sigma1_hat =\n', Sigma1_hat, '\n')

# Class priors
pi0 = N0 / (N0 + N1)
pi1 = N1 / (N0 + N1)
print('Class priors (p.22):')
print('P(Class=0) =', pi0)
print('P(Class=1) =', pi1, '\n')

def log_gaussian_pdf_2d(x, mu, Sigma):
    x = np.atleast_2d(x)
    d = x.shape[1]
    invS = np.linalg.inv(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    diff = x - mu
    qf = np.einsum('...i,ij,...j->...', diff, invS, diff)
    return -0.5 * (d * np.log(2*np.pi) + logdet + qf)

def predict_bayes_2d(x):
    log_p0 = log_gaussian_pdf_2d(x, mu0_hat, Sigma0_hat) + np.log(pi0)
    log_p1 = log_gaussian_pdf_2d(x, mu1_hat, Sigma1_hat) + np.log(pi1)
    return (log_p1 > log_p0).astype(int)

X_2d = np.vstack([X0, X1])
y_2d = np.array([0]*N0 + [1]*N1)
y_pred_2d = predict_bayes_2d(X_2d)
acc_2d = np.mean(y_pred_2d == y_2d)
print('=== Bayes 2D accuracy (p.26 decision rule) ===')
print(f'Accuracy (2D Bayes) = {acc_2d:.3f}\n')

# ============================================================
# PLOTTING : รวม 3 กราฟในหน้าต่างเดียว
#   1) 100D data (feature1 vs feature2)
#   2) 2D raw data
#   3) 2D decision boundary
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---------- กราฟที่ 1 : High-dimensional data (project มา 2D) ----------
ax1 = axes[0]
ax1.scatter(X_A[:, 0], X_A[:, 1], marker='.', s=20, alpha=0.5,
            color='red', label='Class A')
ax1.scatter(X_B[:, 0], X_B[:, 1], marker='.', s=20, alpha=0.5,
            color='blue', label='Class B')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Classwork p.18 (100D → 2D view)')
ax1.grid(True)
ax1.legend()
ax1.set_aspect('equal', 'box')

# ---------- กราฟที่ 2 : 2D Raw data ----------
ax2 = axes[1]
ax2.scatter(X0[:, 0], X0[:, 1], s=15, c='red',  label='Class 0')
ax2.scatter(X1[:, 0], X1[:, 1], s=15, c='blue', label='Class 1')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Raw 2D data (p.19)')
ax2.grid(True)
ax2.legend()
ax2.set_aspect('equal', 'box')

# ---------- กราฟที่ 3 : Decision boundary ----------
ax3 = axes[2]

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
zz = predict_bayes_2d(grid).reshape(xx.shape)

# พื้นหลังเป็น decision region
ax3.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
             alpha=0.3, colors=['red', 'blue'])

# จุดข้อมูลจริง
ax3.scatter(X0[:, 0], X0[:, 1], s=10, c='red',  label='Class 0')
ax3.scatter(X1[:, 0], X1[:, 1], s=10, c='blue', label='Class 1')

# เส้น decision boundary (posterior เท่ากัน)
ax3.contour(xx, yy, zz, levels=[0.5], colors='k', linewidths=2)
ax3.plot([], [], 'k-', label='Decision boundary')  # ไว้ใช้ใน legend

ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_title('Bayes decision boundary (p.26–31)')
ax3.grid(True)
ax3.legend(loc='upper left')
ax3.set_aspect('equal', 'box')

fig.tight_layout()
plt.show()
