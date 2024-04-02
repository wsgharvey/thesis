#%%

import matplotlib.pyplot as plt
import numpy as np
import torch
# %%


# define a Gaussian mixture model
means = [-0.8, 0.3, 0.6]
means = [m-0.34 for m in means]
stds = [0.06, 0.2, 0.05]
weights = [0.1, 0.4, 0.5]

# %%


# plot diffusion process

def get_percentiles(means, stds, weights, sigma, n=1000, percentiles=[0.5]):
    if sigma > 0:
        stds = [(std**2 + sigma**2)**0.5 for std in stds]
    xmin = min(m-3*s for m, s in zip(means, stds))
    xmax = max(m+3*s for m, s in zip(means, stds))
    x = np.linspace(xmin, xmax, n)
    y = np.zeros_like(x)
    for mean, std, weight in zip(means, stds, weights):
        y += weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    y /= y.sum()
    cdf = np.cumsum(y)
    percentiles = np.array(percentiles)
    indices = np.searchsorted(cdf, percentiles)
    return x[indices]


def plot_percentiles_over_process(ax):
    eps = 0.01
    percentiles = np.linspace(eps, 1-eps, 99)
    sigmas = np.linspace(0, 2.1, 20)
    percentiles = np.array([get_percentiles(means, stds, weights, sigma, percentiles=percentiles) for sigma in sigmas])
    for nth_percentile in range(len(percentiles[0])):
        ax.plot(sigmas, percentiles[:, nth_percentile], color='w', alpha=0.1, lw=2)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel('x')


def get_density(means, stds, weights, sigma, xmin, xmax, n):
    if sigma > 0:
        stds = [(std**2 + sigma**2)**0.5 for std in stds]
    x = np.linspace(xmin, xmax, n)
    y = np.zeros_like(x)
    for mean, std, weight in zip(means, stds, weights):
        y += weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return y


def plot_density_over_process(ax):
    xmin, xmax = -5, 5
    sigmas = np.linspace(0, 2.1, 200)
    density = np.array([get_density(means, stds, weights, sigma, xmin, xmax, 1000) for sigma in sigmas]).T
    density = density**0.3  # accentuate the differences between small pdfs
    ax.imshow(density[::-1], interpolation='none', extent=[sigmas[0], sigmas[-1], xmin, xmax], aspect='auto', cmap='Greys_r')


def plot_sde_trajectory(ax, N=100):
    np.random.seed(6)
    # sample x0
    x0s = []
    for _ in range(N):
        component = np.random.choice(len(means), p=weights)
        x0 = np.random.normal(means[component], stds[component])
        x0s.append(x0)
    x0s = np.array(x0s)
    # add noise
    xmin, xmax = -5, 5
    sigmas = np.linspace(0, 2.1, 200)
    dsigma = (sigmas[1:]**2 - sigmas[:-1]**2)**0.5
    xs = [x0s]
    for dsi in dsigma:
        xs.append(xs[-1] + np.random.normal(0, dsi, x0s.shape))
    ax.plot(sigmas, xs, color='r', alpha=0.2)

def get_score(x, means, stds, weights, sigma=0):
    if sigma > 0:
        stds = [(std**2 + sigma**2)**0.5 for std in stds]
    x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    pdf = 0
    for mean, std, weight in zip(*map(torch.tensor, [means, stds, weights])):
        pdf += weight * torch.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    pdf.log().backward()
    return x.grad.item()


def get_ode_trajectory(means, stds, weights):
    # sample x0
    component = np.random.choice(len(means), p=weights)
    x0 = np.random.normal(means[component], stds[component])
    # solve the ODE
    x = [x0]
    ts = np.linspace(0, 2.1, 200)
    t = 0
    dt = ts[1] - ts[0]
    for t in ts[1:]:
        score = get_score(x[-1], means, stds, weights, sigma=t)
        dx = -0.5 * (2*t) * score * dt
        x.append(x[-1] + dx)
    return ts, np.array(x)


def get_ode_trajectory_reverse(xT, means, stds, weights):
    # solve the ODE
    x = [xT]
    ts = np.linspace(2.1, 0, 200)
    dt = ts[0] - ts[1]
    for t in ts[:-1]:
        score = get_score(x[-1], means, stds, weights, sigma=t)
        dx = 0.5 * (2*t) * score * dt
        x.append(x[-1] + dx)
    return ts, np.array(x)


def plot_ode_trajectory(ax, N=3):
    np.random.seed(6)
    for n in range(N):
        ts, xs = get_ode_trajectory(means, stds, weights)
        ax.plot(ts, xs, color='b', alpha=0.3)


def plot_gmm(ax, means, stds, weights, sigma=0, score_x=None, is_log=False, xmin=None, xmax=None):
    if sigma > 0:
        stds = [(std**2 + sigma**2)**0.5 for std in stds]
    xmin = min(m-3*s for m, s in zip(means, stds)) if xmin is None else xmin
    xmax = max(m+3*s for m, s in zip(means, stds)) if xmax is None else xmax
    # plot the pdf in matplotlib
    x = np.linspace(xmin, xmax, 1000)
    y = np.zeros_like(x)
    for mean, std, weight in zip(means, stds, weights):
        y += weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    if is_log:
        y = np.log(y)
    ax.plot(x, y)
    ax.set_xlim(xmin, xmax)
    # fill below even if it's negative
    # ybottom = y.min()-1 if is_log else 0
    ybottom = -12 if is_log else 0
    ytop = y.max() + 0.1*(y.max()-ybottom)
    ax.fill_between(x, y, ybottom, alpha=0.3)
    ax.set_ylim(ybottom, ytop)
    if score_x is not None:
        score = get_score(score_x, means, stds, weights, sigma)
        score_y = np.log(sum([weight * np.exp(-0.5 * ((score_x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)) for mean, std, weight in zip(means, stds, weights)]))
        # draw pointing to right and with score governing its vertical direction
        # starting from score_x, score_y
        arrow_length = 4
        plot_width_over_height = 3.5
        dx = arrow_length / (plot_width_over_height*1 + score**2)**0.5
        print(dx, dx*score)
        # can get bigger arrowhead using head_length and head_width
        ax.annotate('', (score_x+dx, score_y+dx*score), (score_x, score_y), arrowprops=dict(arrowstyle='->', lw=3, color='#0b3b11'), color='#0b3b11')
        ax.scatter(score_x, score_y, color='#0b3b11', marker='x', s=100)

# set font size to same as in my thesis and font to match latex
plt.rcParams.update({'font.size': 11})
plt.rc('text', usetex=True)

# make bottom axes larger
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(5.56, 4.8), gridspec_kw={'height_ratios': [1, 1, 0.4, 2],})
for ax in axes[2]:
    ax.remove()
axes = np.array([axes[0], axes[1], axes[3]])
# conbine three axes on bottom row into a single axis
import matplotlib.gridspec as gridspec
for ax in axes[2]:
    ax.remove()
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 2])
ax = fig.add_subplot(gs[2, :])
plot_density_over_process(ax)
plot_percentiles_over_process(ax)
plot_sde_trajectory(ax, N=100)
# plot_ode_trajectory(ax, N=10)
ax.set_xlim(0, 2.1)
ax.set_xticks([0, 1, 2])
ax.set_ylim(-5, 5)
ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\mathbf{x}_t$', labelpad=-2)
x_ode_vis, y_ode_vis = get_ode_trajectory_reverse(-2.3, means, stds, weights)
ax.plot(x_ode_vis, y_ode_vis, color='b', alpha=1, lw=2)
# get values of y_ode_vis at t=0, t=0.3, t=2
y_ode_mark = [y_ode_vis[np.argmin(np.abs(x_ode_vis - t))] for t in [0, 0.3, 2]]
ax.scatter([0, 0.3, 2], y_ode_mark, color="#0b3b11", marker='x', s=100)
# vertical lines where we plot the marginals
for x in [0.008, 0.3, 2]:
    ax.axvline(x, color='#0b3b11', linestyle='--', lw=2)
# plot the three axes on top row
for ax, sigma in zip(axes[0], [0, 0.3, 2]):
    plot_gmm(ax, means, stds, weights, sigma, xmin=-5, xmax=5)
    ax.set_title(r'$t = ' + str(sigma) +'$', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel(r'$q(\mathbf{x_t})$')
for ax, sigma, y_ode in zip(axes[1], [0, 0.3, 2], y_ode_mark):
    plot_gmm(ax, means, stds, weights, sigma, is_log=True, score_x=y_ode, xmin=-5, xmax=5)
    ax.set_xlabel(r'$\mathbf{x}_t$', labelpad=0)
    ax.set_yticks([])
    ax.set_ylim(-10, 3)
axes[1, 0].set_ylabel(r'$\log~q(\mathbf{x_t})$')
fig.savefig('diffusion_process.pdf', bbox_inches='tight', pad_inches=0)

# %%

