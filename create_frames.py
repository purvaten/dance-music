"""Create visualization for audio+video of dance.

python3.7 record_dance.py

Generate video by entering plots/ folder, and type (frame_rate=10)
ffmpeg -f image2 -r 10 -i %d.png -vcodec mpeg4 -y movie.mp4

Add audio as follows
ffmpeg -i movie.mp4 -i badguy.mp3 -map 0:v -map 1:a -c copy -shortest output.mp4
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import text
import librosa
import pdb
import madmom


LEFT, RIGHT, STAY = -1, 1, 0
mapping = {LEFT: 'L', RIGHT: 'R', STAY: 'S'}


def get_states(start, actions):
    states = []
    s = start
    for a in actions:
        states.append(s)
        s += a
    return states


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

STEPS = len(actions)
positions = get_states(0, actions)

# load song
hop_length = 2048
y, sr = librosa.load(filename)    # default sampling rate 22050
duration = librosa.get_duration(y=y, sr=sr, hop_length=hop_length)
bpm = round(STEPS * 60 / duration)

# get beat information
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm, hop_length=hop_length)
# beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()(filename)
beat_times = proc(act)

# get frequencies
frame_rate = 10
x = beat_times * frame_rate

y = [int(round(x[i]) - int(round(x[i-1]))) for i in range(1, len(x))]
y = np.insert(y, 0, int(round(x[0])))

a = ""
c = 0
for i in range(STEPS):
    a += mapping[actions[i]]
    for _ in range(y[i]):
        show_agent(grid_size=10, t=i, c=c, a=a, position=positions[i], done=False)
        c += 1

print("Done!")
