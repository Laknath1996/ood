import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from matplotlib import animation
from scipy.io import savemat, loadmat

from utils import generate_linear_gaussians

# generate in-distribution data
def generate_in_distribution_data(n):
    X, y = generate_linear_gaussians(n, theta=0, cluster_std=1)
    return X, y

def generate_out_distribution_data(n, theta):
    X, y = generate_linear_gaussians(n, theta=theta, cluster_std=1)
    return X, y

def compute_decision_rule(X, y):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    return clf

def compute_empirical_risk(X, y, clf):
    y_pred = clf.predict(X)
    risk = 1 - np.mean(y_pred == y)
    return risk

## fixed OOD sample size

# n = np.arange(10, 1000, 10)
# n_test = 500
# mu_o = 0
# m = 100
# theta = np.arange(0, 2*np.pi, 0.1)

# X_test, y_test = generate_in_distribution_data(n_test)
# for i in range(len(n)):
#     X, y = generate_in_distribution_data(n[i])
#     risk = []
#     for k in range(len(theta)):
#         X_ood, y_ood = generate_out_distribution_data(m, theta[k])
#         X_comb = np.concatenate((X, X_ood))
#         y_comb = np.concatenate((y, y_ood))
#         X_comb, y_comb = shuffle(X_comb, y_comb)
#         clf = compute_decision_rule(X_comb, y_comb)
#         risk.append(compute_empirical_risk(X_test, y_test, clf))
#     fig, ax = plt.subplots()
#     ax.plot(theta, risk, label='n = {}'.format(n[i]))
#     ax.set_title("n = {}".format(n[i]))
#     ax.set_xlabel(r"$\theta$")
#     ax.set_ylabel(r"$R(h)$")
#     ax.set_ylim([0, 1])
#     plt.show()

## Variable OOD sample size

n = np.arange(10, 1010, 10)
m = np.arange(10, 1010, 10)
n_test = 500
mu_o = 0
theta = np.arange(0, 370, 10)

X_test, y_test = generate_in_distribution_data(n_test)
risk_matrices = {}
for k in range(len(theta)):
    risk = np.zeros((len(n), len(m)))
    for i in range(len(n)):
        for j in range(len(m)):
            X, y = generate_in_distribution_data(n[i])
            X_ood, y_ood = generate_out_distribution_data(m[j], theta[k]/180*np.pi)
            X_comb = np.concatenate((X, X_ood))
            y_comb = np.concatenate((y, y_ood))
            X_comb, y_comb = shuffle(X_comb, y_comb)
            clf = compute_decision_rule(X_comb, y_comb)
            risk[i, j] = compute_empirical_risk(X_test, y_test, clf)
    risk_matrices[theta[k]] = risk

# mdic = {"data": risk_matrices}
# savemat("experiment2_data.mat", mdic)

fig = plt.figure(figsize=(10, 8))
sns.heatmap(risk_matrices[0], vmin=0, vmax=1, cbar_kws={'label': 'Risk'})

def animate(i):
    data = risk_matrices[theta[i]]
    ax = sns.heatmap(data, vmin=0, vmax=1, cbar=False)
    ax.set_title(r"$\theta = {}$".format(theta[i]))
    ax.set_xlabel(r"Out Dist. Sample Size")
    ax.set_ylabel(r"In Dist. Sample Size")
    ax.set_xticks([0, 49, 99])
    ax.set_yticks([0, 49, 99])
    ax.set_xticklabels([10, 500, 1000])
    ax.set_yticklabels([10, 500, 1000])

anim = animation.FuncAnimation(fig, func=animate, frames=5, repeat=False, interval=100)
anim.save('animation.gif', writer='imagemagick', fps=60)
