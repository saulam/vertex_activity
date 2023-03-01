import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_event(X, Y, labels, elev=20, azim=20, img_size=5, dataset=None):
    events = len(X)

    # start plot
    fig = plt.figure(figsize=(events * 18, events * 17))
    fig.patch.set_facecolor('white')

    color = np.array(['#7A88CCC0'])
    edgecolor = '1.0'

    for event in range(events):
        x = X[event].reshape(img_size, img_size, img_size)
        y = Y[event].reshape(img_size, img_size, img_size)

        x[x < dataset.min_charge] = 0
        y[y < dataset.min_charge] = 0

        # fill the detector with reco hits
        detector1 = np.zeros((5, 5, 5), dtype=bool)
        detector_hitcharges1 = x
        colors1 = np.empty((5, 5, 5, 4), dtype=object)

        detector2 = np.zeros((5, 5, 5), dtype=bool)
        detector_hitcharges2 = y
        colors2 = np.empty((5, 5, 5, 4), dtype=object)

        # set colors based on hit charge
        norm = matplotlib.colors.Normalize(vmin=dataset.min_charge, vmax=dataset.max_charge)
        cmap = cm.YlOrRd
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for i in range(detector1.shape[0]):
            for j in range(detector1.shape[1]):
                for k in range(detector1.shape[2]):
                    colors1[i, j, k] = m.to_rgba(detector_hitcharges1[i, j, k])
                    colors2[i, j, k] = m.to_rgba(detector_hitcharges2[i, j, k])

        to_plot = [x, y]
        for i in range(len(to_plot)):
            ax = fig.add_subplot(events, events, i * events + event + 1, projection='3d')

            # voxels volume
            sc = ax.voxels(to_plot[i], facecolors=colors1, edgecolor=edgecolor, alpha=1.0)

            # ax.tick_params(axis='both', which='minor', labelsize=20, length=0)
            ax.set_xlabel('X [cube]', labelpad=50, fontsize=45)
            ax.set_ylabel('Z [cube]', labelpad=50, fontsize=45)
            ax.set_zlabel('Y [cube]', labelpad=50, fontsize=45)

            ini_KE = np.interp(labels[event, -1], dataset.source_range,
                               (dataset.min_E, dataset.max_E)) - dataset.m_proton

            # ticks
            ax.set_title("Total charge: {0:.2f} p.e.\n[KE of {1:.2f} MeV]".format(to_plot[i].sum(), \
                                                                                  ini_KE), fontsize=50)
            ax.set_xticks(np.arange(0.5, 5, 1.), minor=True, length=0, width=0, grid_alpha=0)
            ax.set_xticklabels([str(x) for x in range(1, 6)], minor=True, size=25)
            ax.set_xticklabels([], minor=False)
            ax.set_yticklabels([], minor=False)
            ax.set_yticks(np.arange(0.5, 5, 1.), minor=True, length=0, width=0)
            ax.set_yticklabels([str(x) for x in range(1, 6)], minor=True, size=25)
            ax.set_zticklabels([], minor=False)
            ax.set_zticks(np.arange(0.5, 5, 1.), minor=True)
            ax.set_zticklabels([str(x) for x in range(1, 6)], minor=True, size=25)

            # change camera angle
            ax.view_init(elev=elev, azim=azim)

            ax.grid(False)

    # colorbar
    fig.subplots_adjust(right=0.925)
    cbar_ax = fig.add_axes([0.95, 0.70, 0.01, 0.15])  # left, botton, width, height
    cbar = fig.colorbar(m, cax=cbar_ax, fraction=0.020)
    cbar.set_label('# of p.e.', rotation=90, labelpad=19, fontsize=40)
    cbar.ax.tick_params(labelsize=35)

    plt.show()


def plot_scatter(true, pred, label=None, s=0.01, fontsize=15):
    diff = pred - true
    textstr = '\n'.join((
        r'$\mu \ (reco-true)=%.2f$' % (diff.mean(),),
        r'$\sigma \ (reco-true)=%.2f$' % (diff.std(),)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.scatter(true, pred, s=s)
    plt.xlabel("{0}{1}{2}".format(label[0], "$_{true}$", label[1]), fontsize=fontsize)
    plt.ylabel("{0}{1}{2}".format(label[0], "$_{reco}$", label[1]), fontsize=fontsize)
    plt.text(0.15, 0.84, textstr, transform=plt.gcf().transFigure, fontsize=14,
             verticalalignment='top', bbox=props)
    plt.grid()
    plt.show()


def plot_scatter_len(true, pred, label=None, s=0.01, fontsize=15):
    diff = pred
    textstr = '\n'.join((
        r'$\mu \ (reco-true)=%.2f$' % (diff.mean(),),
        r'$\sigma \ (reco-true)=%.2f$' % (diff.std(),)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.scatter(true, pred, s=s)
    plt.xlabel("particle length [mm]", fontsize=fontsize)
    plt.ylabel("{0}{1}{2}".format(label[0], "$_{reco-true}$", label[1]), fontsize=fontsize)
    plt.text(0.15, 0.84, textstr, transform=plt.gcf().transFigure, fontsize=14,
             verticalalignment='top', bbox=props)
    plt.grid()
    plt.show()


def plot_hist(x, xlim, label=None):
    x = x[x <= xlim]
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (x.mean(),),
        r'$\sigma=%.2f$' % (x.std(),)))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.hist(x, bins=100, color="cornflowerblue")
    plt.ylabel("frequency", fontsize=15)
    plt.xlabel(label, fontsize=15)
    plt.xlim(0, xlim)
    plt.yscale("log")
    plt.ylim(1, 40000)
    plt.yticks((1, 10, 100, 1000, 10000))
    plt.text(0.71, 0.84, textstr, transform=plt.gcf().transFigure, fontsize=14,
             verticalalignment='top', bbox=props)
    plt.show()