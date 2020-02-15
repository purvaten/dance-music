"""Create visualization of recurrence matrix for dance hardcoded to beats (state-action matrix) and music.

python3.7 music_dance.py
"""
from __future__ import print_function
import librosa
import numpy as np
import pdb
import matplotlib.pyplot as plt
import librosa.display
import madmom
from scipy.spatial import distance
from numpy import linalg as LA

LEFT, RIGHT, STAY = -1, 1, 0


def extend_stateaction(state_action, bs, s):
    state_action.insert(0, [0, STAY])
    bs = np.insert(bs, 0, 0)
    state_action_new = []
    for i in range(len(bs)-1):
        for _ in range(int(bs[i+1]-bs[i])):
            state_action_new.append(state_action[i])

    remaining = s - len(state_action_new)
    for i in range(remaining):
        state_action_new.append(state_action[-1])

    return state_action_new



def get_stateaction(start, actions):
    state_action = []
    s = start
    for a in actions:
        state_action.append([s, a])
        s += a
    return state_action


def getval(x, y):
    sx, ax = x
    sy, ay = y
    if sx == sy and ax == ay:
        return 1
    if sx == sy and ax != ay:
        return 0.5
    if sx != sy and ax == ay:
        return 0.5
    return 0


# ****************** SONG-SPECIFIC INPUTS ******************

# # Static example1 - locked away
# filename = 'audio_files/locked_away.mp3'
# actions1 = [RIGHT, RIGHT, LEFT, LEFT] * 8    # If I got locked away..would you still love me the same (x2)
# actions2 = [STAY, STAY, STAY, STAY] * 1      # Pause
# actions3 = [RIGHT, STAY, STAY, LEFT] * 8     # If I judged my life...go on
# actions = actions1 + actions2 + actions3

# Static example2 - bad guy
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
bpm = round(len(actions) * 60 / duration)

# get beat information
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm, hop_length=hop_length)
# beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(filename)
beat_times = proc(act)
print("length of beat_times = ", len(beat_times))
pdb.set_trace()

# ******************TESTING ********************
clicks = librosa.clicks(beat_times, sr=sr, length=len(y))
librosa.output.write_wav('out.mp3', y+clicks, sr)
print("wrote beats to out.mp3")
pdb.set_trace()

# compute music affinity matrix
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, lifter=lifter, hop_length=hop_length)
R = librosa.segment.recurrence_matrix(mfcc, sym=True)
R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sym=True)
R_aff *= 1.0 / R_aff.max()    # normalize it

# how much does a timesecond correspond to in the matrix
s = R.shape[0]
step = R.shape[0] / duration
beat_times_in_matrix_start = np.around(beat_times * step)
beat_times_in_matrix_end = np.around(beat_times_in_matrix_start + step)

# create state action matrix
state_action = get_stateaction(start=0, actions=actions)
state_action = extend_stateaction(state_action, beat_times_in_matrix_start, s)
sa_matrix = np.zeros((s, s))
np.fill_diagonal(sa_matrix, 1)
for i in range(s):
    for j in range(i):
        sa_matrix[i][j] = sa_matrix[j][i] = getval(state_action[i], state_action[j])
sa_aff = np.transpose(sa_matrix)
sa_aff *= 1.0 / sa_aff.max()    # normalize it

# display music affinity matrix along with state action matrix
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
librosa.display.specshow(R_aff, x_axis='time', y_axis='time',
                         hop_length=hop_length, cmap='magma_r')
plt.title('Affinity recurrence')
plt.subplot(1, 2, 2)
librosa.display.specshow(sa_aff, x_axis='time', y_axis='time', hop_length=hop_length)
plt.title('State-action recurrence')
plt.tight_layout()
plt.show()

# Compute similarity between matrices using eiegenvectors
_, music_evectors = LA.eig(R_aff)
_, sa_evectors = LA.eig(sa_aff)

dists = np.array([])
for idx in range(music_evectors.shape[0]):
    mvector = np.real(music_evectors[idx])
    dvector = np.real(sa_evectors[idx])
    dists = np.append(dists, distance.cosine(mvector, dvector))

reward = np.mean(dists) * -1
print("Reward = ", reward)
