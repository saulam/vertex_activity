import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def plot_event(X, Y, labels, elev=20, azim=20, img_size=5, dataset=None):
    nevents = len(X)

    # start plot
    fig = plt.figure(figsize=(nevents * 18, nevents * 17))
    fig.patch.set_facecolor('white')

    color = np.array(['#7A88CCC0'])
    edgecolor = '1.0'

    for event in range(nevents):
        x = X[event].reshape(img_size, img_size, img_size)
        y = Y[event].reshape(img_size, img_size, img_size)

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
            ax = fig.add_subplot(nevents, nevents, i * nevents + event + 1, projection='3d')

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