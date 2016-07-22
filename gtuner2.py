#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
gtuner2.py
(c) Will Roberts  22 July, 2016

Rewrite of the guitar tuner script.

This script uses the Fast Fourier Transform (FFT) to perform frequency
analysis, and sums over a small number of harmonics, to try to
robustly find the fundamental frequency.

Much of this is cribbed from here:

https://gist.github.com/endolith/255291

Other resources used:

https://arxiv.org/pdf/0912.0745.pdf
http://blog.bjornroche.com/2012/07/frequency-detection-using-fft-aka-pitch.html
'''

import re
import time
import matplotlib.pyplot as plt
import numpy
import pyaudio


# ============================================================
#  Digital Signal Processing
# ============================================================

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def build_harmonic_matrix(num_freqs, num_harmonics):
    '''
    Builds a square matrix to sum harmonics.

    Multiplying this matrix with a vector of powers at different
    frequencies (i.e., output of a DFT) gives a similar vector, where
    each coordinate is increased by the sum of its next
    `num_harmonics` harmonics.

    Arguments:
    - `num_freqs`: the size of the matrix to return
    - `num_harmonics`: the number of harmonics to add on to each
      frequency coordinate
    '''
    if num_harmonics < 0:
        num_harmonics = 0
    result = numpy.zeros((num_freqs, num_freqs))
    for harm in range(1, num_harmonics + 2):
        for i in range(int(numpy.ceil(num_freqs / float(harm)))):
            result[i, i * harm] += 1
    return result


# ============================================================
#  Music Theory
# ============================================================

# A4 is defined to be 440.0 Hz
A4 = 440.

# map names of notes onto number of semitone offsets
NOTES_TO_SEMITONES = {'A': 0,
                      'A#': 1,
                      'Bb': 1,
                      'B': 2,
                      'C': -9,
                      'C#': -8,
                      'Db': -8,
                      'D': -7,
                      'D#': -6,
                      'Eb': -6,
                      'E': -5,
                      'F': -4,
                      'F#': -3,
                      'Gb': -3,
                      'G': -2,
                      'G#': -1,
                      'Ab': -1}

# reverse of the mapping above
SEMITONES_TO_NOTES = dict((y,x) for (x,y) in NOTES_TO_SEMITONES.items()
                          if not x.endswith('b'))

STANDARD_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

def interpret_note(note):
    '''
    Interprets a note representation by breaking it into its
    constituent parts.

    >>> interpret_note('A4')
    ('A', 4)
    >>> interpret_note('Bb-1')
    ('Bb', -1)

    Arguments:
    - `note`:
    '''
    # interpret the note string
    match = re.match(r'(?P<letter>[A-Za-z#]{1,2})(?P<octave>-?[0-9]{1,2})', note)
    if not match:
        raise Exception('Could not interpret note "{}"'.format(note))
    letter = match.group('letter')
    octave = int(match.group('octave'))
    if len(letter) < 1 or letter[0] not in 'ABCDEFG':
        raise Exception('Could not interpret note "{}"'.format(letter))
    if len(letter) > 1 and (len(letter) > 2 or letter[1] not in '#b'):
        raise Exception('Could not interpret note "{}"'.format(letter))
    if letter not in NOTES_TO_SEMITONES:
        raise Exception('Could not interpret note "{}"'.format(letter))
    return letter, octave

def find_freq(note):
    '''
    Converts a note into a frequency in Hertz.

    https://en.wikipedia.org/wiki/Musical_note#Note_frequency_.28hertz.29

    >>> find_freq('A4')
    440.0
    >>> find_freq('C5')
    523.25113060119725
    >>> find_freq('F4')
    349.22823143300388
    >>> find_freq('E4')
    329.62755691286992
    >>> find_freq('B3')
    246.94165062806206
    >>> find_freq('G3')
    195.99771799087463
    >>> find_freq('D3')
    146.83238395870379
    >>> find_freq('A2')
    110.0
    >>> find_freq('E2')
    82.406889228217494

    Arguments:
    - `note`: a string representing a note, like A4, Ab4 or C#-1
    '''
    letter, octave = interpret_note(note)
    # calculate the number of semitones away from A4
    offset = NOTES_TO_SEMITONES[letter] + (octave - 4) * 12
    return numpy.exp2(offset / 12.) * A4

def find_closest_note(freq):
    '''
    Finds the note which is closest to the given frequency.

    >>> find_closest_note(440)
    'A4'
    >>> find_closest_note(523.3)
    'C5'
    >>> find_closest_note(349.2)
    'F4'
    >>> find_closest_note(329.6)
    'E4'
    >>> find_closest_note(246.9)
    'B3'
    >>> find_closest_note(196.0)
    'G3'
    >>> find_closest_note(146.8)
    'D3'
    >>> find_closest_note(110)
    'A2'
    >>> find_closest_note(82.4)
    'E2'

    Arguments:
    - `freq`: a frequency given in Hz
    '''
    # f = 2^(n/12) x 440 Hz
    # f / 440 Hz = 2^(n/12)
    # n = 12 * log_2(f / 440 Hz)
    offset = 12 * numpy.log2(freq / A4)
    # quantise to an integer number
    ioffset = int(numpy.round(offset))
    # ioffset needs to be in the range -9 to 2 (inclusive)
    octave = (ioffset + 9) // 12
    qoffset = (ioffset + 9) % 12 - 9
    return '{}{}'.format(SEMITONES_TO_NOTES[qoffset], 4 + octave)


# ============================================================
#  Main Program
# ============================================================

# audio data is read in in chunks (buffering)
CHUNK       = 1024
# record 32-bit floating point signed mono 8 KHz
FORMAT      = pyaudio.paFloat32
CHANNELS    = 1
SAMPLE_RATE = 8000

# our desired frequency resolution is 0.5 Hz
FREQ_RESOLUTION = 0.5 # Hz

# this is the number of samples we need to be able to calculate
# frequencies with sufficient accuracy.
MIN_TAPE_LENGTH = int(SAMPLE_RATE / FREQ_RESOLUTION)

# we want to update the screen twice a second
SCREEN_UPDATE_DELAY = 0.5 # s

# pre-calculate a matrix to allow summing harmonics
# this one uses the base frequency, plus harmonics 2, 3, and 4
HARMONIC_MATRIX = build_harmonic_matrix(MIN_TAPE_LENGTH // 2 + 1, 3)

# pre-calculate the hamming window
HAMMING_WINDOW = numpy.hamming(MIN_TAPE_LENGTH)

# we want to plot frequencies up to 2 KHz
MAX_PLOT_FREQ = 2000 # Hz

# the number of frequencies to plot
NUM_PLOTTED_FREQS = int(MAX_PLOT_FREQ / FREQ_RESOLUTION) + 1

# the x labels of the frequencies to plot
FREQ_PLOT_XLABELS = np.arange(NUM_PLOTTED_FREQS) * FREQ_RESOLUTION

# these are hand-calibrated values for determining what is "loud" to
# my microphone
RMS_RANGES = (('g', 0, 1.0),
              ('y', 1.0, 10.),
              ('r', 10., 20.))

def main():

    # Step 1: program set up: open audio device, set up variables,
    # etc.

    audio = pyaudio.PyAudio()
    stream = audio.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = SAMPLE_RATE,
                        input = True,
                        frames_per_buffer = CHUNK)
    tape = numpy.array([], dtype=numpy.float32)
    last_update_time = time.time()
    relevant_strings = set([interpret_note(x)[0] for x in STANDARD_TUNING])
    plt.ion()

    # Step 2: loop

    print 'Press Ctrl-C to quit ...'
    try:
        while True:

            # Step 3: record some audio, tack it onto the end of our
            # "tape"

            try:
                data = stream.read(CHUNK)
            except IOError:
                # ignore spurious IOErrors while reading audio
                continue
            # convert data into numpy vector
            samples = numpy.fromstring(data, dtype='float32')
            # store running data into tape
            tape = numpy.append(tape, samples)

            # Step 4: if we have enough data, and it is time to update the
            # screen

            if (MIN_TAPE_LENGTH <= len(tape) and
                SCREEN_UPDATE_DELAY < time.time() - last_update_time):

                last_update_time = time.time()

                # Step 5: select the last two seconds of data and window it

                selected = tape[-MIN_TAPE_LENGTH:] * HAMMING_WINDOW

                # Step 6: take the FFT

                freqs = abs(numpy.fft.rfft(selected))

                # Step 7: sum the harmonics
                # Note: endolith does a *product* of harmonics; we
                # just add them all up (without coefficients)

                freqs = HARMONIC_MATRIX.dot(freqs)

                # Step 8: find the maximum frequency

                max_freq_idx = numpy.argmax(abs(freqs))
                est_max_freq_idx = parabolic(abs(freqs), max_freq_idx)[0]
                est_max_freq = SAMPLE_RATE * est_max_freq_idx / MIN_TAPE_LENGTH
                closest_note = find_closest_note(est_max_freq)
                print('{:.2f} Hz ({})'.format(est_max_freq, closest_note))

                # Step 9: plot things

                # calculate the current sound power
                rms_power = np.sqrt((freqs ** 2).mean())
                # find the closest string to the note we've identified
                # (closest_note)
                closest_string = 'G4'

                plt.clf()
                plt.subplot(311)

                # Text and RMS power
                plt.subplot(312)
                plt.axis('off')
                plt.xlim(xmin=0, xmax=RMS_RANGES[-1][-1])
                plt.ylim(ymin=0, ymax=1)
                plt.text(0.5, 0.5,
                         'Main Freq: {:.1f} Hz\n'
                         'Closest Note: {}\n'
                         'String: {}\n'
                         'RMS Power: {:.1f}'.format(
                             est_max_freq, closest_note, closest_string, rms_power))
                color = 'g'
                for (rms_color, rms_min, _rms_max) in RMS_RANGES:
                    if rms_min <= rms_power:
                        color = rms_color
                plt.barh(bottom=0.2, height=0.2, width=rms_power, color=color)

                # Frequency spectrum
                plt.subplot(313)
                plt.plot(FREQ_PLOT_XLABELS, np.log(freqs[:NUM_PLOTTED_FREQS]))
                plt.ylabel('Power')
                plt.xlabel('Frequency (Hz)')
                plt.draw()
                plt.pause(0.01)

                # Step 10: shorten the tape if needed

                tape = tape[-MIN_TAPE_LENGTH:]

            pass

    except KeyboardInterrupt:
        pass

    print
    print "* done recording"

    stream.close()
    audio.terminate()