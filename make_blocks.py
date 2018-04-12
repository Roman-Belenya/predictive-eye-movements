import numpy as np
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
