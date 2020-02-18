"""Visualize recurrence matrix for music and dance steps and final movie output.

python3.7 music_dance.py

Generate video by entering plots/ folder, and type
ffmpeg -f image2 -r 306/28.40498866213152 -i %d.png -vcodec mpeg4 -y movie.mp4

Add audio as follows
ffmpeg -i movie.mp4 -i badguy.mp3 -map 0:v -map 1:a -c copy -shortest output.mp4
"""
from __future__ import print_function
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import madmom
from scipy.spatial import distance
from numpy import linalg
from pylab import text

import pdb
import sys


LEFT, RIGHT, STAY = -1, 1, 0
mapping = {LEFT: 'L', RIGHT: 'R', STAY: 'S'}


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """Return True if matrix a is symmetric, else False."""
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def show_agent(grid_size, t, c, a, position, done):
    """Save grid of agent in current position at dance step count t."""
    loc = position + 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if done:
        plt.plot(loc, 0, 'ro', markersize=25)
    else:
        plt.plot(loc, 0, 'bo', markersize=25)
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    frame1.set_xlim([0, grid_size])
    plt.xticks(np.arange(0, grid_size + 1, 1.0))
    plt.yticks(np.arange(-1, 2, 1.0))

    ax.axhline(y=.25)
    ax.axhline(y=-.25)
    for i in range(grid_size + 1):
        ax.axvline(x=i, ymin=0.4, ymax=0.6)

    text(0.1, 0.9, "t = " + str(t), ha='center', va='center', transform=ax.transAxes, fontsize=20)
    # text(0.5, 0.8, "actions = " + str(a), ha='center', va='center', transform=ax.transAxes, fontsize=5)
    fig.savefig('plots/' + str(c) + '.png')


def extend_stateaction(state_action, bs, s, pos):
    """Extend state action list based acc to the exact beat times proportions.

    This is only used for checking results of hard-coded dance.
    Otherwise, state_action list will be obtained during training.
    """
    # set initialization
    state_action.insert(0, [pos, STAY])

    state_action_new = []
    actions = []
    positions = []
    for i in range(len(bs)):
        for j in range(int(bs[i]-bs[i-1])):
            if j == 0:
                state_action_new.append(state_action[i])
                actions.append(state_action[i][1])
                positions.append(state_action[i][0])
            else:
                next_state = state_action_new[-1][0] + state_action_new[-1][1]
                state_action_new.append([next_state, STAY])
                actions.append(STAY)
                positions.append(next_state)

    remaining = s - len(state_action_new)
    for i in range(remaining):
        if i == 0:
            state_action_new.append(state_action[-1])
            actions.append(state_action[-1][1])
            positions.append(state_action[-1][0])
        else:
            next_state = state_action_new[-1][0] + state_action_new[-1][1]
            state_action_new.append([next_state, STAY])
            actions.append(STAY)
            positions.append(next_state)

    return state_action_new, actions, positions


def get_stateaction(start, actions):
    """Return list of (state, action) tuples for each step."""
    state_action = []
    s = start
    for a in actions:
        state_action.append([s, a])
        s += a
    return state_action


def getval(x, y):
    """Return value in matrix based on if action matches or not."""
    sx, ax = x
    sy, ay = y
    if ax == ay:
        return 0
    return 1

# ****************** SONG-SPECIFIC INPUTS ******************

# # Static example1 - locked away -- Reward =  0.9121797821332441, Mean dists =  0.9962751198687801

# filename = 'audio_files/locked_away.mp3'
# actions1 = [RIGHT, RIGHT, LEFT, LEFT] * 8    # If I got locked away..would you still love me the same (x2)
# actions2 = [STAY, STAY, STAY, STAY] * 1      # Pause
# actions3 = [RIGHT, STAY, STAY, LEFT] * 8     # If I judged my life...go on
# actions = actions1 + actions2 + actions3

# Static example2 - bad guy -- Reward =  0.9066821520641596, Mean dists =  1.0029223391277662
filename = 'audio_files/badguy.mp3'
actions1 = [RIGHT, RIGHT, LEFT, LEFT] * 4
actions2 = [RIGHT, LEFT, RIGHT, LEFT] * 4
actions3 = [RIGHT, STAY, STAY, LEFT] * 8
actions = actions1 + actions2 + actions3

# load song
y, sr = librosa.load(filename)    # default sampling rate 22050
duration = librosa.get_duration(y=y, sr=sr)

# parameters
lifter = 0
hop_length = 2048
n_mfcc = 20

# get beat information
proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(filename)
beat_times = proc(act)

# compute music affinity matrix
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, lifter=lifter, hop_length=hop_length)
R = librosa.segment.recurrence_matrix(mfcc, sym=True)    # already normalized in 0-1
R = np.transpose(R)
R = R + np.zeros(R.shape)

# how much does a timesecond correspond to in the matrix
step = R.shape[0] / duration
beat_times_in_matrix_start = np.around(beat_times * step)

# create state action matrix, and get all actions too
s = R.shape[0]
state_action = get_stateaction(start=0, actions=actions)
state_action, actions, positions = extend_stateaction(state_action, beat_times_in_matrix_start, s, 0)

sa_aff = np.zeros((s, s))
np.fill_diagonal(sa_aff, 0)
for i in range(s):
    for j in range(i):
        sa_aff[i][j] = sa_aff[j][i] = getval(state_action[i], state_action[j])
if sa_aff.max() != 0:
    sa_aff *= 1.0 / sa_aff.max()    # normalize it

# display music affinity matrix along with state action matrix
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
# librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=hop_length, cmap='magma_r')
plt.imshow(R)
plt.title('Affinity recurrence')
plt.subplot(1, 2, 2)
# librosa.display.specshow(sa_aff, x_axis='time', y_axis='time', hop_length=hop_length, cmap='magma_r')
plt.imshow(sa_aff)
plt.title('State-action recurrence')
plt.tight_layout()
plt.show()

# Compute similarity between matrices using eiegenvectors
_, music_evectors = linalg.eig(R)
_, sa_evectors = linalg.eig(sa_aff)

dists = np.array([])
for idx in range(music_evectors.shape[0]):
    mvector = np.real(music_evectors[idx])
    dvector = np.real(sa_evectors[idx])
    dists = np.append(dists, distance.cosine(mvector, dvector))

reward = 1 / (np.mean(dists) + 0.1)
print("Reward = ", reward)
print("Mean dists = ", np.mean(dists))

sys.exit()

# Create dance video
print("Creating dance video")
a = ""
c = 0
for i in range(len(actions)):
    a += mapping[actions[i]]
    show_agent(grid_size=10, t=i, c=c, a=a, position=positions[i], done=False)
    c += 1

print("Done!")
