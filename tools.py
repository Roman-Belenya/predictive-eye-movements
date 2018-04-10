import numpy as np
import matplotlib.pyplot as plt

def dispersion(eyex, eyez, win):

    d = ( max(eyex[win[0]:win[1]]) - min(eyex[win[0]:win[1]]) ) + \
        ( max(eyez[win[0]:win[1]]) - min(eyez[win[0]:win[1]]) )

    return d


def distance(p1, p0):

    p0 = np.array(p0); p1 = np.array(p1)
    d = np.sqrt(np.sum( (p1 - p0)**2, axis = 0 ))

    return d


def velocity(x, y, z):

    vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)

    return vel


def check_accuracy(trial):

    eye_x = trial.data['averageXeye'] * 100
    eye_z = trial.data['averageZeye'] * 100

    centre_x = trial.data['objectX'][0] * 100
    centre_z = trial.data['objectZ'][0] * 100

    error_x = trial.data['errorX'] * 100
    error_y = trial.data['errorY'] * 100
    error_z = trial.data['errorZ'] * 100

    dist = trial.data['totalError'] * 100
    mean_dist = np.mean(dist)
    median_dist = np.median(dist)

    r1 = 1
    r2 = 0.5
    theta = np.arange(0, 2.01 * np.pi, np.pi / 100)
    x = r1 * np.cos(theta) + centre_x
    z = r1 * np.sin(theta) + centre_z

    fig = plt.figure()
    fig.subplots_adjust(wspace=0.05, top=1, right=0.97, left=0.03, bottom=0)

    ax1 = fig.add_subplot(131, aspect = 'equal')
    ax1.plot(eye_x, eye_z, 'b.', alpha = 0.6)
    ax1.plot(x, z, 'r-')
    ax1.plot(r2 * np.cos(theta) + centre_x, r2 * np.sin(theta) + centre_z, ':', color = '#FF8D0D')
    ax1.plot(np.mean(eye_x), np.mean(eye_z), 'r.', markersize = 15)
    ax1.plot(np.median(eye_x), np.median(eye_z), '.', color = '#FF8D0D', markersize = 15)
    ax1.plot(centre_x, centre_z, 'k+', markersize = 10)
    ax1.set_xlim([centre_x - 2, centre_x + 2])
    ax1.set_ylim([centre_z - 2, centre_z + 2])

    ax2 = fig.add_subplot(132)
    ax2.plot(dist, 'b-')
    ax2.axhline(1, color = 'r')
    ax2.axhline(mean_dist, color = 'r', linestyle = ':', label = 'Mean')
    ax2.axhline(median_dist, color = '#FF8D0D', linestyle = ':', label = 'Median')
    ax2.set_ylim([0, 5])
    x0, x1 = ax2.get_xlim()
    y0, y1 = ax2.get_ylim()
    asp = (x1 - x0) / (y1 - y0)
    ax2.set_aspect(asp)
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(error_x, 'b-', label = 'Error X')
    ax3.plot(error_z, 'g-', label = 'Error Z')
    ax3.plot(error_y, 'y-', label = 'Error Y')
    ax3.axhline(1, color = 'r')
    ax3.axhline(-1, color = 'r')
    ax3.axhline(0, color = 'k', linestyle = ':')
    ax3.set_ylim([-2.5, 2.5])
    x0, x1 = ax3.get_xlim()
    y0, y1 = ax3.get_ylim()
    asp = (x1 - x0) / (y1 - y0)
    ax3.set_aspect(asp)
    ax3.legend()

    plt.show()
