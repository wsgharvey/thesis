import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ground_truth_path = 'ground_truth.png'
hmc_samples = [f"hmc/{i}.png" for i in range(3)]
retrieval_samples = [f"retrieval/{i}.png" for i in range(3)]
gan_samples = [f"gan/{i}.png" for i in range(3)]

observed_sequence = [(88, 63), (88, 42), (181, 130)]
att_dim = 16
DOWNSAMPLING = 2
original_img_dim = 224
img_dim = original_img_dim // DOWNSAMPLING

def load_img(path, downsampled=True):
    img = Image.open(path).resize((original_img_dim, original_img_dim))
    if downsampled:
        img = img.resize((img_dim, img_dim))
    return np.array(img)

def plot_red_squares(ax):
    for r, c in observed_sequence:
        r = r / DOWNSAMPLING
        c = c / DOWNSAMPLING
        sqr_dim = att_dim // DOWNSAMPLING
        rect = Rectangle((c-0.5, r-0.5), sqr_dim, sqr_dim, linewidth=0.5,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)

def plot_glimpses(ax, path):
    img = load_img(path, False)
    glimpses = np.concatenate([img[r:r+att_dim, c:c+att_dim, :]
                               for r, c in observed_sequence],
                              axis=0)
    for h in range(0, att_dim*len(observed_sequence), att_dim):
        rect = Rectangle((-0.5, h-0.5), att_dim, att_dim, linewidth=1,
                         edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.imshow(glimpses)

n_samples = 3
width_ratios = [1, 0.02, 0.35, 0.4,] + [1, 0.02, 0.35, 0.15,]*n_samples
height_ratios = [1]*3
fig, axes = plt.subplots(3, len(width_ratios),
                         gridspec_kw={'width_ratios': width_ratios,
                                      'height_ratios': height_ratios,
                                      'wspace': 0,
                                      'hspace': 0.01},
                         figsize=(5, 2.7))

# plot true image
ground_truth = load_img(ground_truth_path)
for ax in axes[:, 0]:
    ax.imshow(ground_truth)
    plot_red_squares(ax)

# get blanked out version
for ax in axes[:, 2]:
    plot_glimpses(ax, ground_truth_path)

for axes_row, sample_paths in zip(axes[:, -4*n_samples:],
                                  [gan_samples,
                                   hmc_samples,
                                   retrieval_samples]):
    for main_ax, glimpse_ax, path in zip(axes_row[::4],
                                         axes_row[2::4],
                                         sample_paths):
        try:
            img = load_img(path)
            main_ax.imshow(img)
            plot_red_squares(main_ax)
            plot_glimpses(glimpse_ax, path)
        except FileNotFoundError:
            pass

for ax in np.array(axes).reshape(-1):
    ax.xaxis.set_visible(False)
    plt.setp(ax.spines.values(), visible=False)
    ax.tick_params(left=False, labelleft=False)
    ax.patch.set_visible(False)

fontsize=10
axes[0, 0].set_title('True Image', fontsize=fontsize)
axes[0, 4].set_title('Samples...', fontsize=fontsize)
def label_row(i, label):
    axes[i, 0].set_ylabel(label, rotation=0,
                          fontsize=fontsize, labelpad=35,
                          va='center')
label_row(0, 'Conditional\nGAN')
label_row(1, 'HMC')
label_row(2, 'Image\nretrieval')

plt.tight_layout()
fig.savefig('samples.pdf', bbox_inches='tight')
"3799   64002   94878  104409  118852  169102  219452  221488 307144  311038  339647  351555  353483  358180  366181  412394 427095  429279  457267  458260  518862  580900  603204  608000 624441  633073  641078  644039  666737  685963  688864  698469 748067  801555  840423  888478  918020  932939  988774 1053632 1075470 1089321 1096636 1097488 1097682 1146558 1195588 1195644 1222972 1225988 1231051 1270367 1298642 1300596 1318922 1319672 1333116 1341113 1388961 1396481 1397773 1413073 1426943 1467504"
