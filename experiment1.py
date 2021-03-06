# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from matplotlib import animation

# define support functions

# generate in-distribution data
def generate_in_distribution_data(n, mu_o):
    X = np.concatenate((1*np.random.randn(int(n/2))+ (mu_o-1), 1*np.random.randn(int(n/2))+ (mu_o+1)))
    y = np.concatenate((np.zeros((int(n/2), )), np.ones(int(n/2), )))
    X, y = shuffle(X, y)
    return X, y

def generate_out_distribution_data(n, mu_ood):
    X = np.concatenate((1*np.random.randn(int(n/2))+ (mu_ood-1), 1*np.random.randn(int(n/2))+ (mu_ood+1)))
    y = np.concatenate((np.zeros((int(n/2), )), np.ones(int(n/2), )))
    X, y = shuffle(X, y)
    return X, y

def compute_decision_rule(X, y):
    n = len(y)
    mu_hat = (1/n)*np.sum(X - (-1)**y)
    return mu_hat

def compute_empirical_risk(X, y, mu_hat):
    y_pred = (X > mu_hat).astype('int')
    risk = 1 - np.mean(y_pred == y)
    return risk
    
# %%

## fixed OOD sample size

# n = np.arange(10, 1000, 10)
# n_test = 500
# mu_o = 0
# m = 100
# epsilon = 0.3125
# mu_ood = np.arange(0, 10, 0.1)

# X_test, y_test = generate_in_distribution_data(n_test, mu_o)
# for i in range(len(n)):
#     X, y = generate_in_distribution_data(n[i], mu_o)
#     risk = []
#     mu_hats = []
#     for k in range(len(mu_ood)):
#         X_ood, y_ood = generate_out_distribution_data(m, mu_ood[k])
#         X_comb = np.concatenate((X, X_ood))
#         y_comb = np.concatenate((y, y_ood))
#         X_comb, y_comb = shuffle(X_comb, y_comb)
#         mu_hat = compute_decision_rule(X_comb, y_comb)
#         risk.append(compute_empirical_risk(X_test, y_test, mu_hat))
#         mu_hats.append(mu_hat)
#     c_n = (n[i] + m)/m * epsilon
#     fig, ax = plt.subplots()
#     ax.plot(mu_ood, risk, label='n = {}'.format(n[i]))
#     ax.set_title("n = {}, c(n) = {}".format(n[i], c_n))
#     ax.set_xlabel("$|\mu_0 - \mu_{ood}|$")
#     ax.set_ylabel("$R(h)$")
#     ax.set_ylim([0, 0.6])
#     ax.axhline(y=0.16, color='r')
#     ax.axvline(x=c_n, color='k')
#     plt.show()

## Variable OOD sample size

n = np.arange(10, 1010, 10)
m = np.arange(10, 1010, 10)
n_test = 500
mu_o = 0
mu_ood = np.arange(-10, 10, 0.1)
epsilon = 0.3125

X_test, y_test = generate_in_distribution_data(n_test, mu_o)
risk_matrices = {}
for k in range(len(mu_ood)):
    risk = np.zeros((len(n), len(m)))
    for i in range(len(n)):
        for j in range(len(m)):
            X, y = generate_in_distribution_data(n[i], mu_o)
            X_ood, y_ood = generate_out_distribution_data(m[j], mu_ood[k])
            X_comb = np.concatenate((X, X_ood))
            y_comb = np.concatenate((y, y_ood))
            X_comb, y_comb = shuffle(X_comb, y_comb)
            clf = compute_decision_rule(X_comb, y_comb)
            risk[i, j] = compute_empirical_risk(X_test, y_test, clf)
    risk_matrices[mu_ood[k]] = risk

print(risk_matrices)

fig = plt.figure(figsize=(10, 8))
sns.heatmap(risk_matrices[mu_ood[0]], vmin=0, vmax=1, cbar_kws={'label': 'Risk'})

def animate(i):
    data = risk_matrices[mu_ood[i]]
    ax = sns.heatmap(data, vmin=0, vmax=1, cbar=False)
    ax.set_title("$|\mu_0 - \mu_{ood}| = $" + str(mu_ood[i]))
    ax.set_xlabel(r"Out Dist. Sample Size")
    ax.set_ylabel(r"In Dist. Sample Size")
    ax.set_xticks([0, 49, 99])
    ax.set_yticks([0, 49, 99])
    ax.set_xticklabels([10, 500, 1000])
    ax.set_yticklabels([10, 500, 1000])

anim = animation.FuncAnimation(fig, func=animate, frames=len(mu_ood), repeat=False, interval=100)
anim.save('animation.gif', writer='imagemagick', fps=60)


# %%
