from datetime import datetime
from collections import OrderedDict
import re
import numpy as np
import tools as tls
import matplotlib.pyplot as plt; plt.ion()
import os
import glob
import sys
import cPickle as pickle
import warnings


class Trial(object):

    counter = 0

    def __init__(self, filename = None):

        self.increment_counter()

        self.trialname = None
        self.type = None
        self.datetime = None
        self.capture_period = None
        self.data = None
        self.exclude = False
        self.empty = False
        self.filename = filename

        if not filename:
            self.number = self.get_counter()
            self.exclude = True
            self.empty = True
            return

        self.read_file()
        self.number = self.get_trial_number()

        if self.type != 'accuracy':
            assert self.number == self.get_counter()
            self.compute_variables()


    def __getattr__(self, var_name):
        try:
            return self.data[var_name]
        except:
            raise AttributeError('{} is not a variable in Trial'.format(var_name))


    @classmethod
    def reset_counter(cls):
        cls.counter = 0

    @classmethod
    def increment_counter(cls):
        cls.counter += 1

    @classmethod
    def get_counter(cls):
        return cls.counter


    def read_file(self):

        with open(self.filename, 'rb') as f:

            lines = [line.rstrip().split('\t') for line in f]

            self.trialname = lines[2][0]
            self.type = re.search('[a-z]+', self.trialname).group()

            dt = lines[2][1] + ' ' + lines[2][2]
            self.datetime = datetime.strptime(dt, '%d-%m-%Y %H:%M:%S:%f')

            self.framerate = float(lines[3][0])
            self.capture_period = float(lines[4][0])

            colheaders = [l.replace(' ', '') for l in lines[8]]
            data_by_rows = [map(float, l) for l in lines[9:]]
            data_by_columns = zip(*data_by_rows)
            data_by_columns = [np.array(x) for x in data_by_columns]

            self.data = OrderedDict(zip(colheaders, data_by_columns))


    def get_trial_number(self):

        name = os.path.split(self.filename)[1]
        number = name.split('_')[0]
        try:
            return int(number)
        except:
            return number


    def exclude_trial(self):
        self.exclude = True


    def compute_variables(self):

        self.data['eye_x'] = np.mean([self.LEyeInterX, self.REyeInterX], axis = 0)
        self.data['eye_y'] = np.mean([self.LEyeInterY, self.REyeInterY], axis = 0)
        self.data['eye_z'] = np.mean([self.LEyeInterZ, self.REyeInterZ], axis = 0)

        # Velocity unit is metres/frame
        self.data['wrist11_vel'] = tls.velocity(self.Wrist11x, self.Wrist11y, self.Wrist11z)
        self.data['wrist12_vel'] = tls.velocity(self.Wrist12x, self.Wrist12y, self.Wrist12z)

        self.fixations = self.get_fixations()


    def check_markers_consistency(self):

        index = tls.distance(
            [ self.Index7x, self.Index7y, self.Index7z ],
            [ self.Index8x, self.Index8y, self.Index8z ])

        thumb = tls.distance(
            [ self.Thumb9x, self.Thumb9y, self.Thumb9z ],
            [ self.Thumb10x, self.Thumb10y, self.Thumb10z ])

        wrist = tls.distance(
            [ self.Wrist11x, self.Wrist11y, self.Wrist11z ],
            [ self.Wrist12x, self.Wrist12y, self.Wrist12z ])

        eyes = tls.distance(
            [ self.LEyeInterX, self.LEyeInterY, self.LEyeInterZ ],
            [ self.REyeInterX, self.REyeInterY, self.REyeInterZ ])

        return index, thumb, wrist, eyes


    def get_fixations(self, disp_th = 0.01, dur_th = 0.1):

        win = [0, int(dur_th * self.framerate)]
        fixations = []
        eyex = self.eye_x; eyey = self.eye_y; eyez = self.eye_z

        while win[1] < len(eyex):

            d = tls.dispersion(eyex, eyez, win)
            if d <= disp_th:

                while d <= disp_th and win[1] < len(eyex):
                    win[1] += 1
                    d = tls.dispersion(eyex, eyez, win)

                if win[1] != len(eyex):
                    win[1] -= 1

                fixations.append([
                    win[0], # start frame
                    win[1], # end frame
                    (win[1] - win[0] + 1.0),# / self.framerate, # duration
                    np.mean(eyex[win[0]:win[1]]), # centre x
                    np.mean(eyez[win[0]:win[1]]), # centre z
                    ])

                win = [win[1] + 1, win[1] + int(self.framerate * dur_th)]

            else:
                win = [x + 1 for x in win]

        return fixations


    def find_fixation(self, time):

        # Find last fixation that starts before time
        fixs = filter(lambda x: x[0] <= time, self.fixations)
        try:
            return fixs[-1]
        except:
            warnings.warn('No fixation that starts before frame {}, trial {}'.format(time, self.get_trial_number()))
            return None




class Participant(object):

    def __init__(self, name, dirname):

        Trial.reset_counter()

        self.name = name
        self.dirname = dirname
        self.organise()

        self.exclude = False
        self.condition = self.identify_condition()
        self.best_index = None
        self.best_thumb = None
        self.best_wrist = None

        self.check_trials_order()


    def __len__(self):

        n = len( filter(lambda x: not x.exclude, self.iter_trials()) )
        return n


    def len_trials(self, kind, block = 'both'):

        n = len( filter(lambda x: x.type == kind, self.iter_trials(block)) )
        return n


    def iter_trials(self, block = 'both'):

        if block == 'both':
            to_iter = self.block1 + self.block2
        else:
            to_iter = getattr(self, block)

        for trial in to_iter:
            yield trial


    def __getattr__(self, trial):

        try:
            kind, ind = re.match('([at])([0-9]+)', trial).groups()
            ind = int(ind)

            assert 0 < ind < 121

            if kind == 't':
                return (self.block1 + self.block2)[ind-1]
            elif kind == 'a':
                return self.accuracies[ind-1]

        except:
            raise AttributeError('Invalid trial name: {}'.format(trial))


    def organise(self):

        files = glob.glob(os.path.join(self.dirname, '*.exp'))

        trial_files = [n for n in files if not 'accuracy' in n.lower()]
        trial_files.sort(key = tls.get_trial_int)

        for i, trialname in enumerate(trial_files):
            n = tls.get_trial_int(trialname)
            if n != i+1:
                trial_files.insert(i, None)

        trial_files = tls.none_pad(trial_files)
        trials = [Trial(n) for n in trial_files]
        self.block1 = trials[:30]
        self.block2 = trials[30:]

        acc_files = [n for n in files if 'accuracy' in n.lower()]
        acc_files.sort()
        self.accuracies = [Trial(n) for n in acc_files]


    def identify_condition(self):

        l = self.len_trials('leftward', 'block2')
        r = self.len_trials('rightward', 'block2')

        if l > r:
            return 'Left'
        elif r > l:
            return 'Right'
        else:
            warnings.warn('Impossible to identify condition for {}'.format(self.name))
            return None


    def set_markers(index, thumb, wrist, overwrite = False):

        assert index in ['Index7', 'Index8'], 'Invalid marker {}'.format(index)
        assert thumb in ['Thumb9', 'Thumb10'], 'Invalid marker {}'.format(thumb)
        assert wrist in ['Wrist11', 'Wrist12'], 'Invalid marker {}'.format(wrist)

        already_set = any( [self.best_index, self.best_thumb, self.best_wrist] )
        if already_set and not overwrite:
            warnings.wars('Markers for {} already set'.format(self.name))
            return

        self.best_index = index
        self.best_thumb = thumb
        self.best_wrist = wrist


    def exclude_participant(self):
        self.exclude = True


    def check_trials_order(self):

        if self.condition == 'Left':
            template = tls.read_template_file('./mm_scripts/left_template.txt')
        elif self.condition == 'Right':
            template = tls.read_template_file('./mm_scripts/right_template.txt')
        else:
            return

        for trial, info in zip(self.iter_trials(), template):
            if trial.empty:
                trial.type = info[1]

            assert trial.number == info[0], '{}, {}'.format(trial.number, info[0])
            assert trial.type == info[1], '{}, {}'.format(trial.type, info[1])

        print 'all trials are in order'


    def check_accuracy(self, which = 'both'):

        if which == 'both':
            to_check = self.accuracies
        else:
            to_check = self.__getattr__(which)

        for acc in to_check:
            tls.check_accuracy(acc)


    def check_marker(self, marker = 'index', get_all = False):

        if get_all:
            l = ['index', 'thumb', 'wrist', 'eyes']
        else:
            l = [marker]

        for m in l:
            tls.check_marker(self.iter_trials(), m)


    def check_fixations(self, block = 'both'):
        tls.check_fixations(self.iter_trials(block))


class Experiment(object):

    def __init__(self):

        self.participants = OrderedDict()


    def __iter__(self):

        for name, participant in self.participants.items():
            yield name, participant


    def __getattr__(self, name):
        try:
            return self.participants[name.lower()]
        except:
            raise AttributeError('Participant {} does not exist'.format(name))


    def __len__(self):
        n = filter(lambda x: not x.exclude, self)
        return n


    def read_participants_data(self, data_dir, skip_existing = False):

        dirs = os.listdir(data_dir)
        for dir in dirs:
            print dir
            if dir in self.participants.keys() and skip_existing:
                continue
            p = Participant(dir.lower(), os.path.join(data_dir, dir))
            self.participants[p.name] = p


    def save_data(self, filename = None):

        if not filename:
            dt = datetime.now()
            filename = './saved_' + dt.strftime('%d-%b-%Y_%H-%M-%S') + '.pkl'

        with open(filename, 'ab') as f:
            for participant in self.participants.values():
                pickle.dump(participant, f, pickle.HIGHEST_PROTOCOL)


    def load_data(self, filename = None):

        if not filename:
            all_saved = glob.glob('./*.pkl')
            all_saved.sort(key = tls.sort_saved_by_date)
            filename = all_saved[-1] # most recent
            print '---> from {}:'.format(filename)

        for participant in tls.pickled_participants(filename):
            self.participants[participant.name] = participant
            print '\t' + participant.name

        return self

