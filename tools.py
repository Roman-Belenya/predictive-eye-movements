import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def chunks(items, n):
    # https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(items), n):
        yield items[i:i + n]


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
        pos = 0.607
    elif dim == 'z':
        idx = 4
        pos = 0.322
    else:
        return

    fig, ax = plt.subplots()

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

    ax.axvline(100, color = 'k', linestyle = ':', alpha = 0.7, linewidth = 1)
    ax.axvline(300, color = 'k', linestyle = ':', alpha = 0.7, linewidth = 1)

    ax.axhline(pos, color = 'k', linestyle = '-', alpha = 0.7, linewidth = 0.5)
    ax.axhline(pos + 0.02, color = 'k', linestyle = ':', alpha = 0.7, linewidth = 1)
    ax.axhline(pos - 0.02, color = 'k', linestyle = ':', alpha = 0.7, linewidth = 1)

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


def make_blocks():

    np.random.seed(1)
    acc = '''Prefs "Roman_Accuracy"\nBiofeedback\nExport "{}_Roman_Accuracy.exp"\n\n'''
    left = '''Prefs "Roman_Left"\nBiofeedback\nExport "{}_Roman_Left.exp"\n\n'''
    right = '''Prefs "Roman_Right"\nBiofeedback\nExport "{}_Roman_Right.exp"\n\n'''

    n1 = 30
    p1 = 0.5
    n2 = 90
    p2 = 0.8

    # Unbiased script
    n_left = n_right = int(round(n1 * p1))
    seq = [left] * n_left + [right] * n_right
    for i in range(100):
        np.random.shuffle(seq)

    with open('unbiased.txt', 'w') as f:
        f.write(acc.format('a0'))
        for i, line in enumerate(seq):
            f.write(line.format(i+1))


    n_left = int(round(p2 * n2))
    n_right = int(round((1-p2) * n2))

    ratio = n_left / n_right + 1
    left_bias_seq = []
    right_bias_seq = []

    for i in range(n2 / ratio):

        part_l = [left] * ratio
        part_r = [right] * ratio

        n = np.random.choice(range(ratio))
        part_l[n] = right
        part_r[n] = left
        left_bias_seq.extend(part_l)
        right_bias_seq.extend(part_r)


    # Left bias script
    with open('left_bias.txt', 'w') as f:
        f.write(acc.format('a1'))
        for i, line in enumerate(left_bias_seq):
            f.write(line.format(i+n1+1))

    # Right bias script
    with open('right_bias.txt', 'w') as f:
        f.write(acc.format('a1'))
        for i, line in enumerate(right_bias_seq):
            f.write(line.format(i+n1+1))




def within_range(xy, lim_x, lim_y):
    return lim_x[0] < xy[0] < lim_x[1] and lim_y[0] < xy[1] < lim_y[1]



class AnalyseFixations(object):

    def __init__(self, exp, timerange, index = None):

        self.exp = exp
        self.timerange = timerange
        self.index = index

        self.n_total = 0
        self.n_excluded = 0
        self.data = None

        self.get_fixations()

    @property
    def excluded_percent(self):
        return float(self.n_excluded * 100) / self.n_total

    def __iter__(self):

        for group in self.data:
            for part in group:
                for fix in part:
                    yield fix


    def get_fixations(self):

        left = [[], [], [], []]
        right = [[], [], [], []]

        total_n = 0.0
        excluded_n = 0.0

        for participant in self.exp:
            if participant.exclude:
                continue

            for i, part in participant.iter_parts():
                for trial in part:

                    if trial.exclude:
                        continue

                    self.n_total += 1

                    fs = trial.find_fixations(self.timerange)

                    if not fs:
                        self.n_excluded += 1
                        continue

                    if self.index >= len(fs): # uncertain here: pos/neg index
                        self.n_excluded += 1
                        continue

                    if self.index is not None:
                        fs = [fs[self.index]]

                    if participant.condition == 'Left':
                        left[i].extend(fs)
                    elif participant.condition == 'Right':
                        right[i].extend(fs)

        self.data = [left, right]


    def remove_outliers(self, offscreen = True, std = True):

        if not (offscreen or std):
            return self

        new_data = []

        for group in self.data:
            new_group = []
            for part in group:

                orig_len = len(part)

                if offscreen:

                    lim_x = (0.337, 0.874)
                    lim_z = (0.173, 0.473)

                    part = [x for x in part if within_range(x[3:5], lim_x, lim_z)]

                if std:

                    avg_x, avg_z = np.mean(part, axis = 0)[3:5]
                    std_x, std_z = np.std(part, axis = 0)[3:5]

                    lim_x = (avg_x - 2*std_x, avg_x + 2*std_x)
                    lim_z = (avg_z - 2*std_z, avg_z + 2*std_z)


                    part = [x for x in part if within_range(x[3:5], lim_x, lim_z)]

                new_group.append(part)
                self.n_excluded += orig_len - len(part)

            new_data.append(new_group)
        self.data = new_data

        return self


    def make_histograms(self, title = None, idx = 3):

        fig, axs = plt.subplots(nrows = 2, ncols = 4, sharex = True, sharey = True, figsize = (10,7))

        for i, row in enumerate(axs):
            for k, ax in enumerate(row):

                d = [x[idx] for x in self.data[i][k]]

                ax.hist(d, bins = 30)
                ax.set_xlim([0.607 - 0.05, 0.607 + 0.05])
                ax.set_title('part {}, n = {}'.format(k, len(self.data[i][k])))

                avg = np.mean(d)
                med = np.median(d)
                std = np.std(d)
                stderr = std / np.sqrt(len(d))

                ax.axvspan(0.607-0.02, 0.607+0.02, color = 'k', alpha = 0.1)
                ax.axvline(avg, color = 'r', linewidth = 1)
                ax.axvspan(avg - stderr, avg + stderr, color = 'r', alpha = 0.1)

        plt.suptitle(title)



    def make_scatters(self, title = None):

        fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize = (10,7))

        for i, row in enumerate(axs):
            for k, ax in enumerate(row):

                xs = [x[3] for x in self.data[i][k]]
                ys = [x[4] for x in self.data[i][k]]

                ax.scatter(xs, ys, s = 5, alpha = 0.5, edgecolors = 'none')
                ax.axis('square')
                ax.set_xlim([0.607-0.05, 0.607+0.05])
                ax.set_ylim([0.322-0.05, 0.322+0.05])

                ax.set_title('part {}, n = {}'.format(k, len(self.data[i][k])))

                avg = np.mean(self.data[i][k], axis = 0)[3:5]
                med = np.median(self.data[i][k], axis = 0)[3:5]
                std = np.std(self.data[i][k], axis = 0)[3:5]

                ax.plot(avg[0], avg[1], 'rx')
                ax.errorbar(avg[0], avg[1], xerr = std[0], yerr = std[1], color = 'k', linewidth = 1)
                ax.add_patch(patches.Rectangle([0.607-0.01, 0.322-0.01], 0.02, 0.02,
                    color = 'k', alpha = 0.1))
                # ax.add_patch(patches.Ellipse(avg, 2*std[0], 2*std[1],
                #     color = 'r', alpha = 0.1))

        plt.suptitle(title)
