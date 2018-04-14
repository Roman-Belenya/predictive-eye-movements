import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import datetime
import cPickle as pickle

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


def none_pad(arr, length = 120):

    while len(arr) < length:
        arr = np.append(arr, None)

    return arr


def get_trial_int(str):

    name = os.path.split(str)[1]
    return int(name.split('_')[0])


def sort_saved_by_date(path):
    name = os.path.split(path)[1]
    date = name.lstrip('saved_')
    date = date.rstrip('.pkl')
    dt = datetime.datetime.strptime(date, '%d-%b-%Y_%H-%M-%S')

    return dt


def read_template_file(filename):
    with open(filename, 'rb') as f:
        lines = [line.rstrip().split() for line in f.readlines()]

    for line in lines:
        line[0] = int(line[0])

    return lines



def pickled_participants(filename):

    # https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


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

    plt.gcf().suptitle(trial.number)

    plt.show()


def check_marker(trials, marker):

    if marker == 'index':
        types = ('Index7', 'Index8')
    elif marker == 'thumb':
        types = ('Thumb9', 'Thumb10')
    elif marker == 'wrist':
        types = ('Wrist11', 'Wrist12')
    elif marker == 'eyes':
        types = ('LEyeInter', 'REyeInter')

    fig, axs = plt.subplots(1, 2, sharex = True)

    for trial in trials:

        if trial.exclude:
            continue

        for i, t in enumerate(types):
            for dim, col in [('x', 'r'), ('y', 'g'), ('z', 'b')]:
                if marker == 'eyes':
                    dim = dim.upper()
                axs[i].plot(trial.data[t + dim], color = col, linewidth = 0.5, alpha = 0.2)

    axs[0].set_title(types[0])
    axs[1].set_title(types[1])
    plt.show()


def check_fixations(trials, dim = 'x'):

    if dim == 'x':
        idx = 3
        pos = trials[0].ObjectX[0]
    elif dim == 'z':
        idx = 4
        pos = trials[0].ObjectZ[0]
    else:
        return

    fig, ax = plt.subplots()

    ax.axvline(100, color = 'k', linestyle = ':', alpha = 0.5)
    ax.axvline(300, color = 'k', linestyle = ':', alpha = 0.5)
    ax.axhline(pos, color = 'k', linestyle = ':', alpha = 0.5)

    for trial in trials:

        if trial.exclude:
            continue

        columns = zip(*trial.fixations)
        # plt.plot(columns[3], columns[4], 'k.', alpha = 0.5)
        ax.scatter(columns[0], columns[idx],
            marker = '.',
            c = 'b',
            alpha = 0.5,
            s = np.array(columns[2]) * 0.5)
        ax.plot(columns[0], columns[idx], 'b-', alpha = 0.1, linewidth = 0.3)

    plt.show()


def increment_file_names(dir_name, by = 1):

    files = glob.glob(os.path.join(dir_name, '*.exp'))
    new_dir = os.path.join(dir_name, 'incremented')
    os.mkdir(new_dir)

    for file in files:

        path, name = os.path.split(file)
        if name.startswith('a'):
            continue
        name_split = name.split('_')
        name_split[0] = str(int(name_split[0]) + by)

        new_name = os.path.join(new_dir, '_'.join(name_split))

        print '{} ---> {}'.format(file, new_name)
        shutil.copyfile(file, new_name)


def rename_manually_exported_trials(dir, activities_file):

    files = glob.glob(os.path.join(dir, '*.exp'))

    with open(activities_file) as f:
        activities = [line.rstrip().split() for line in f]

    for file in files:
        filename = os.path.split(file)[-1]
        activity_name = filename.rstrip('.exp')

        # if 'accuracy' in activity_name:
        #     continue

        activity_info, = filter(lambda x: x[1] == activity_name, activities)

        if 'leftward' in activity_name:
            cond = 'Left'
        elif 'rightward' in activity_name:
            cond = 'Right'
        elif 'accuracy' in activity_name:
            cond = 'Accuracy'

        new_name = '_'.join([activity_info[0], 'Roman', cond]) + '.exp'
        print '{} ---> {}'.format(filename, new_name)
        os.rename(file, os.path.join(dir, new_name))
