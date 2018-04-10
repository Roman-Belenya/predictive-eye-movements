from datetime import datetime
from collections import OrderedDict
import re
import numpy as np
import tools as tls
import matplotlib.pyplot as plt

class Trial(object):

    def __init__(self, filename):

        self.filename = filename
        self.read_file()

        if self.type != 'accuracy':
            self.compute_variables()
            self.get_fixations()

        self.exclude = False


    def read_file(self):

        with open(self.filename, 'rb') as f:

            lines = [line.rstrip().split('\t') for line in f]

            self.trialname = lines[2][0]
            self.type = re.search('[a-z]+', self.trialname).group()

            dt = lines[2][1] + ' ' + lines[2][2]
            self.datetime = datetime.strptime(dt, '%m-%d-%Y %H:%M:%S:%f')

            self.framerate = float(lines[3][0])
            self.capture_period = float(lines[4][0])

            colheaders = [l.replace(' ', '') for l in lines[8]]
            data_by_rows = [map(float, l) for l in lines[9:]]
            data_by_columns = zip(*data_by_rows)
            data_by_columns = [np.array(x) for x in data_by_columns]

            self.data = OrderedDict(zip(colheaders, data_by_columns))


    def compute_variables(self):

        self.data['eye_x'] = np.mean([self.data['LEyeInterX'], self.data['REyeInterX']], axis = 0)
        self.data['eye_y'] = np.mean([self.data['LEyeInterY'], self.data['REyeInterY']], axis = 0)
        self.data['eye_z'] = np.mean([self.data['LEyeInterZ'], self.data['REyeInterZ']], axis = 0)

        # Velocity unit is metres/frame
        self.data['wrist11_vel'] = tls.velocity(self.data['Wrist11x'], self.data['Wrist11y'], self.data['Wrist11z'])
        self.data['wrist12_vel'] = tls.velocity(self.data['Wrist12x'], self.data['Wrist12y'], self.data['Wrist12z'])


    def check_markers_consistency(self):

        index = tls.distance(
            [ self.data['Index7x'], self.data['Index7y'], self.data['Index7z'] ],
            [ self.data['Index8x'], self.data['Index8y'], self.data['Index8z'] ])

        thumb = tls.distance(
            [ self.data['Thumb9x'], self.data['Thumb9y'], self.data['Thumb9z'] ],
            [ self.data['Thumb10x'], self.data['Thumb10y'], self.data['Thumb10z'] ])

        wrist = tls.distance(
            [ self.data['Wrist11x'], self.data['Wrist11y'], self.data['Wrist11z'] ],
            [ self.data['Wrist12x'], self.data['Wrist12y'], self.data['Wrist12z'] ])

        eyes = tls.distance(
            [ self.data['LEyeInterX'], self.data['LEyeInterY'], self.data['LEyeInterZ'] ],
            [ self.data['REyeInterX'], self.data['REyeInterY'], self.data['REyeInterZ'] ])

        return index, thumb, wrist, eyes


    def dispersion(self, eyex, eyez, win):

        d = ( max(eyex[win[0]:win[1]]) - min(eyex[win[0]:win[1]]) ) + \
            ( max(eyez[win[0]:win[1]]) - min(eyez[win[0]:win[1]]) )

        return d


    def get_fixations(self, disp_th = 0.01, dur_th = 0.1):

        win = [0, int(dur_th * self.framerate)]
        fixations = []
        eyex = self.data['eye_x']; eyey = self.data['eye_y']; eyez = self.data['eye_z']

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
                    (win[1] - win[0] + 1.0) / self.framerate, # duration
                    np.mean(eyex[win[0]:win[1]]), # centre x
                    np.mean(eyez[win[0]:win[1]]), # centre z
                    ])

                win = [win[1] + 1, win[1] + int(self.framerate * dur_th)]

            else:
                win = [x + 1 for x in win]

        self.fixations = fixations


    def get_var(self, name):
        if name in self.colheaders:
            idx = colheaders.index(name)
            return self.data[idx]



class Participant(object):

    def __init__(self, name, dirname):

        self.name = name
        self.dirname = dirname

        self.exclude = False


    def organise(self):

        trials = [n for n in os.listdir(self.dirname) if not n.startswith('a')]
        accuracies = [n for n in os.listdir(self.dirname) if n.startswith('a')]

        trials.sort(key = lambda x: int(x.split('_')[0]))
        accuracies.sort()
