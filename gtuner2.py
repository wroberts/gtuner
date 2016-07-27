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
http://dsp.stackexchange.com/questions/1317/is-there-an-algorithm-for-finding-a-frequency-without-dft-or-fft

Other implementations:

https://www.snip2code.com/Snippet/36379/PyTuner---Small-Python-Command-Line-Tune
- contains low-pass FIR filter

Future reading (autocorrelation):

http://stackoverflow.com/a/5045834/1062499
http://recherche.ircam.fr/equipes/pcm/cheveign/pss/2002_JASA_YIN.pdf
http://fivedots.coe.psu.ac.th/~montri/Research/Publications/iscit2003_pda.pdf

Further reading (phase-locked loops):

http://arachnoid.com/phase_locked_loop/

Other software:

http://www.nongnu.org/lingot/
'''

import itertools
import re
import sys
import time
import matplotlib.pyplot as plt
import numpy
import pyaudio
import scipy.signal


# ============================================================
#  Digital Signal Processing
# ============================================================

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    it1, it2 = itertools.tee(iterable)
    next(it2, None)
    return itertools.izip(it1, it2)

def parabolic(vec, argmax):
    '''
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    vec is a vector and argmax is an index for that vector.

    Returns (vertx, verty), the coordinates of the vertex of a
    parabola that goes through point argmax and its two neighbors.

    Example:
    Defining a vector vec with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: vec = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(vec, argmax(vec))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    '''
    vertx = 1/2. * (vec[argmax-1] - vec[argmax+1]) / (vec[argmax-1] - 2 * vec[argmax] +
                                                      vec[argmax+1]) + argmax
    verty = vec[argmax] - 1/4. * (vec[argmax-1] - vec[argmax+1]) * (vertx - argmax)
    return (vertx, verty)

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

NOTE_STRING_REGEX = re.compile(r'(?P<letter>[A-Za-z#]{1,2})(?P<octave>-?[0-9]{1,2})')

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
    match = NOTE_STRING_REGEX.match(note)
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

TAPE_COORDINATES = numpy.arange(MIN_TAPE_LENGTH)

# we want to update the screen twice a second
SCREEN_UPDATE_DELAY = 0.5 # s

# pre-calculate a matrix to allow summing harmonics
# this one uses the base frequency, plus harmonics 2, 3, and 4
NUM_HARMONICS = 3
HARMONIC_MATRIX = build_harmonic_matrix(MIN_TAPE_LENGTH // 2 + 1, NUM_HARMONICS)

# pre-calculate the hamming window
HAMMING_WINDOW = numpy.hamming(MIN_TAPE_LENGTH)

# we want to plot frequencies up to 2 KHz
MAX_PLOT_FREQ = 2000 # Hz

# the number of frequencies to plot
NUM_PLOTTED_FREQS = int(MAX_PLOT_FREQ / FREQ_RESOLUTION) + 1

# the x labels of the frequencies to plot
FREQ_PLOT_XLABELS = numpy.arange(NUM_PLOTTED_FREQS) * FREQ_RESOLUTION

# these are hand-calibrated values for determining what is "loud" to
# my microphone
RMS_RANGES = (('g', sys.float_info.min, 0.0),
              ('y', 0.0, 2.3025850929940459),
              ('r', 2.3025850929940459, 2.9957322735539909))

NUM_SIDEBANDS = 9

def sideband_energies(samples, target_freq, sideband_width_loghz, num_sidebands, num_harmonics):
    '''
    Compute the Fourier response in `samples` along the `target_freq`
    and a number of "sidebands" alongside it.

    Arguments:
    - `samples`: a vector of length MIN_TAPE_LENGTH
    - `target_freq`: a frequency in Hertz
    - `sideband_width_loghz`: the total width (given in log Hz) of all
      the sidebands
    - `num_sidebands`: the total number of sidebands
    - `num_harmonics`: the number of harmonics above the fundamental
      to add on to each Fourier series
    '''
    log_target_freq = numpy.log2(target_freq)
    sidebands = numpy.logspace(log_target_freq - sideband_width_loghz / 2,
                               log_target_freq + sideband_width_loghz / 2,
                               num_sidebands,
                               base=2.)
    # compute the fourier series for each of the sidebands
    fourier = numpy.exp(numpy.outer(sidebands, TAPE_COORDINATES) * -2j * numpy.pi / SAMPLE_RATE)
    if num_harmonics < 0:
        num_harmonics = 0
    for harm in range(2, 2 + num_harmonics):
        fourier += numpy.exp(numpy.outer(sidebands * harm, TAPE_COORDINATES) *
                             -2j * numpy.pi / SAMPLE_RATE)
    return numpy.abs(fourier.dot(samples).real)

def compute_tuning_vars(tuning):
    '''
    Computes several variables relevant to fixing the tuning of a
    guitar string.

    Arguments:
    - `tuning`:
    '''
    relevant_strings = set([interpret_note(x)[0] for x in tuning])
    # log frequencies of the strings we're using in our
    # tuning, all on octave 4
    string_freqs_loghz = dict((x, numpy.log2(find_freq('{}4'.format(x))))
                              for x in relevant_strings)
    # find the minimum distance between relevant strings in log
    # frequency space
    freqs_loghz = numpy.array(sorted(string_freqs_loghz.values()))
    sideband_width_loghz = min((y - x) % 1 for (x, y) in
                               itertools.islice(
                                   pairwise(itertools.cycle(freqs_loghz)),
                                   len(freqs_loghz)))
    return sideband_width_loghz, string_freqs_loghz

def find_main_freq(freqs):
    '''
    Identifies the main frequency from the vector of frequencies `freqs`.

    Arguments:
    - `freqs`:
    '''
    max_freq_idx = numpy.argmax(abs(freqs))
    est_max_freq_idx = parabolic(abs(freqs), max_freq_idx)[0]
    est_max_freq = SAMPLE_RATE * est_max_freq_idx / MIN_TAPE_LENGTH
    closest_note = find_closest_note(est_max_freq)
    return est_max_freq, closest_note

def find_target_freq(main_freq, closest_string):
    '''
    Given that we think we're tuning `closest_string`, what frequency
    should we be aiming for?

    Arguments:
    - `main_freq`: the main frequency identified in the recorded sound
    - `closest_string`: the name of a string (like 'A', or 'D#'); a
      note without an octave
    '''
    main_freq_loghz = numpy.log2(main_freq)
    string4_freq_loghz = numpy.log2(find_freq('{}4'.format(closest_string)))
    target_freq = numpy.exp2(string4_freq_loghz -
                             numpy.round(string4_freq_loghz - main_freq_loghz))
    return target_freq

def find_closest_string(string_freqs_loghz, main_freq):
    '''
    Find the closest string to the note we've identified
    (closest_note).

    Arguments:
    - `string_freqs_loghz`: a dict mapping string names (e.g., 'A')
      onto frequencies (in log Hz, in octave 4)
    - `main_freq`:
    '''
    # put main_freq into octave 4 (between find_freq('C4') and find_freq('C5'))
    # we work in log space
    # numpy.log2(find_freq('C4')) = 8.03
    # numpy.log2(find_freq('C5')) = 9.03
    # numpy.log2(main_freq) = 10.001
    c4_loghz = numpy.log2(find_freq('C4'))
    main_freq_oct4_loghz = (numpy.log2(main_freq) - c4_loghz) % 1 + c4_loghz
    # collect the strings and log frequencies we're searching (in
    # order of ascending frequency)
    search_items = sorted(string_freqs_loghz.items(), key=lambda x: x[1])
    # wrap around: place the lowest frequency string, one octave up,
    # at the end of the search list
    search_items.extend([(x, y + 1) for (x, y) in search_items[:1]])
    closest_string = min([((main_freq_oct4_loghz - y) ** 2, x) for (x, y) in
                          search_items])[1]
    return closest_string

def draw_ui(main_freq, closest_note, selected, freqs,
            string_freqs_loghz, sideband_width_loghz):
    '''
    Draw information to the screen.

    Arguments:
    - `main_freq`:
    - `closest_note`:
    - `selected`:
    - `freqs`:
    - `string_freqs_loghz`:
    - `sideband_width_loghz`: the total width of the sideband
      frequency space, in log Hz
    '''
    # calculate the current sound power
    rms_power = numpy.log(numpy.sqrt((freqs ** 2).mean()))

    # find the closest string to the note we've identified
    # (closest_note)
    closest_string = find_closest_string(string_freqs_loghz, main_freq)
    # find the frequency that the string should be
    target_freq = find_target_freq(main_freq, closest_string)

    # compute the sideband energies
    sb_energies = sideband_energies(selected,
                                    target_freq,
                                    sideband_width_loghz,
                                    NUM_SIDEBANDS,
                                    NUM_HARMONICS)
    sb_bestidx = numpy.argmax(sb_energies)

    plt.clf()
    # sideband energies
    plt.subplot(311)
    plt.gca().xaxis.set_ticks([]) # remove x ticks
    plt.gca().yaxis.set_ticks([]) # remove y ticks
    bars = plt.bar(left=numpy.arange(NUM_SIDEBANDS)+0.05,
                   height=sb_energies,
                   width=.9,
                   color='k')
    # color the most powerful narrowband specially
    if sb_bestidx == (NUM_SIDEBANDS - 1) // 2:
        bars[sb_bestidx].set_facecolor('g')
    else:
        bars[sb_bestidx].set_facecolor('r')

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
                 main_freq, closest_note, closest_string, rms_power))
    color = 'g'
    for (rms_color, rms_min, _rms_max) in RMS_RANGES:
        if rms_min <= rms_power:
            color = rms_color
    plt.barh(bottom=0.2, height=0.2, width=rms_power, color=color)

    # Frequency spectrum
    plt.subplot(313)
    plt.plot(FREQ_PLOT_XLABELS, numpy.log(freqs[:NUM_PLOTTED_FREQS]))
    plt.ylabel('Power (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.draw()
    plt.pause(0.01)

def main():
    '''
    Main program.  A guitar tuner running inside matplotlib.
    '''

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
    sideband_width_loghz, string_freqs_loghz = compute_tuning_vars(STANDARD_TUNING)
    # low-pass filter with frequency cutoff of 1.5 kHz
    lopass_filter = scipy.signal.firwin(1001, 1500. / (0.5 * SAMPLE_RATE), window='hamming')
    plt.ion()
    plt.gcf().canvas.set_window_title('Guitar Tuner')

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

                # Step 5a: low-pass filter up to 329.60 x 4 = 1318.4 Hz
                selected = scipy.signal.lfilter(lopass_filter, 1, selected)

                # Step 6: take the FFT

                freqs = abs(numpy.fft.rfft(selected))

                # Step 7: sum the harmonics
                # Note: endolith does a *product* of harmonics; we
                # just add them all up (without coefficients)

                freqs = HARMONIC_MATRIX.dot(freqs)

                # Step 8: find the maximum frequency

                main_freq, closest_note = find_main_freq(freqs)
                #print('{:.2f} Hz ({})'.format(main_freq, closest_note))

                # Step 9: plot things

                draw_ui(main_freq, closest_note, selected, freqs,
                        string_freqs_loghz, sideband_width_loghz)

                # Step 10: shorten the tape if needed

                tape = tape[-MIN_TAPE_LENGTH:]

    except KeyboardInterrupt:
        pass

    print
    print "* done recording"

    stream.close()
    audio.terminate()

if __name__ == '__main__':
    main()
