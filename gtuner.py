#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
gtuner.py
(c) 2012 Will Roberts

A guitar tuner application built using pylab and pyaudio.  Run from
the command line and press Ctrl-C to quit.
'''

import matplotlib.pyplot as plt
import numpy
import pyaudio
import sys

# chunks of 512 samples gives a frequency resolution of about 80 Hz;
# 1024 gives 40 Hz; 2048 gives about 20 Hz
CHUNK      = 2048
# record 16-bit signed mono 44.1 KHz
FORMAT     = pyaudio.paInt16
CHANNELS   = 1
RATE       = 44100
# these are hand-calibrated values for determining what is "loud" to
# my microphone
RMS_RANGES = (('g', 0, 15000),
              ('y', 15000, 75000),
              ('r', 75000, 150000))
# update the display every 10 secs (1/6 minute)
DISPLAY_UPDATE = 0.16666
# http://en.wikipedia.org/wiki/Piano_key_frequencies
TUNING = {
    82.4069: 'E',
    110.0:   'A',
    146.832: 'D',
    195.998: 'G',
    246.942: 'B',
    329.6276: 'E',
}
# compute the minimum harmonic distance between two strings
NUM_NARROWBANDS = 9
STRING_EXPANSION_COEFFS = 2 ** (numpy.arange(5) - 1.) # 0.5, 1, 2, 4, 8
EXPANDED_STRING_VALUES = sorted(set(
    (TUNING.keys() * STRING_EXPANSION_COEFFS.reshape(
            (len(STRING_EXPANSION_COEFFS),1))).reshape(
        len(STRING_EXPANSION_COEFFS) * len(TUNING))))
MIN_BETWEEN_STRING_DIST = min(abs(
    (numpy.append([0],numpy.log(EXPANDED_STRING_VALUES)) -
     numpy.append(numpy.log(EXPANDED_STRING_VALUES),[0]))[1:-1])) / 2.

def find_tuning(freq):
    '''
    Find the string in TUNING which best matches the given frequency.
    '''
    best_err    = None
    best_string = None
    best_sfreq  = None
    for (sfreq, string) in TUNING.iteritems():
        # find the best octave for the given string
        octave_offset = int(round(numpy.log(freq / sfreq) / numpy.log(2.)))
        # compute the squared error of the string's frequency in that
        # octave with the most powerful frequency.
        sqerr = (numpy.log(freq) - numpy.log(sfreq * 2 ** octave_offset)) ** 2
        if best_err is None or sqerr < best_err:
            best_string = string
            best_err = sqerr
            best_sfreq = sfreq * 2 ** octave_offset
    return best_string, best_sfreq

def narrowband_spec(center_freq, num_bands = NUM_NARROWBANDS):
    '''
    Determines the centre frequencies (in Hz) of a set of bands,
    centered on center_freq, and extending out harmonically on both
    sides halfway to the next nearest string.
    '''
    band_range = (num_bands - 1) / 2
    output = []
    for band in range(-band_range, num_bands - band_range):
        output.append(center_freq *
                      numpy.exp(MIN_BETWEEN_STRING_DIST * band / band_range))
    return numpy.array(output)

def fourier_response(samples, freq):
    '''
    Returns the Fourier response of the given samples to the given
    frequency.
    '''
    return abs(numpy.exp(
            numpy.arange(samples.size) *
            freq * -2j * numpy.pi / RATE).dot(samples).real)

def narrowband_responses(frame, narrowband_freqs):
    '''
    Returns the Fourier responses of the frame for each of the
    frequency bands centered on a frequency listed in
    narrowband_freqs.
    '''
    # X (cycles / sample) = FREQ (cycles / second) / RATE (samples / second)
    responses = [fourier_response(frame, freq) for freq in narrowband_freqs]
    # this computes the discrete fourier transform for frequency 0.444
    # 0.444 means 0.444 cycles per sample
    # numpy.exp(numpy.array(range(v.size)) * 0.444 * -2j * numpy.pi).dot(v).real
    return numpy.array(responses)

def main():
    '''
    Main function.
    '''
    audio = pyaudio.PyAudio()

    stream = audio.open(format = FORMAT,
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        frames_per_buffer = CHUNK)

    # set interactive matplotlib for Mac OS X and clear the figure
    plt.ion()
    plt.clf()
    # this doesn't work
    fig = plt.gcf()
    fig.canvas.set_window_title('Guitar Tuner')

    # frequencies (x-axis labels)
    freqs = numpy.fft.fftfreq(CHUNK, 1./RATE)[:(CHUNK / 2)]

    print "* recording"
    # accumulate frequencies and data
    cum = numpy.zeros(len(freqs))
    frame = numpy.array([], dtype=numpy.int16)
    i = 0
    print 'Press Ctrl-C to quit ...'
    try:
        while True:
            try:
                data = stream.read(CHUNK)
            except IOError:
                # ignore spurious IOErrors while reading audio
                continue
            # convert data into numpy vector
            samples = numpy.fromstring(data, dtype='int16')
            # store running data into frame
            frame = numpy.append(frame, samples)
            # the absolute values of the real components (rfft for
            # real input values computes the series only up to the
            # nyquist frequency)
            spec = abs(numpy.fft.rfft(samples).real)
            if len(spec) < len(cum):
                spec = numpy.append(spec,
                                    numpy.array([0] * (len(cum) - len(spec))))
            if len(cum) < len(spec):
                spec = spec[:len(cum)]
            # store into the accumulator
            cum += spec
            # display every so often
            if i > 0 and i % int(round(RATE / CHUNK * DISPLAY_UPDATE)) == 0:
                plt.clf()
                # choose the main frequency
                main_freq = freqs[numpy.argmax(cum)]
                # find the string which best fits the main frequency
                string, string_freq = find_tuning(main_freq)
                # show the narrowband frequency analysis
                nb_axes = plt.subplot(311)
                nb_axes.get_xaxis().set_ticks([]) # remove x ticks
                nb_axes.get_yaxis().set_ticks([]) # remove y ticks
                nb_responses = narrowband_responses(
                    frame, narrowband_spec(string_freq))
                bars = plt.bar(left=numpy.arange(NUM_NARROWBANDS)+0.05,
                               height=nb_responses,
                               width=.9,
                               color='k')
                # color the most powerful narrowband specially
                nb_bestidx = numpy.argmax(nb_responses)
                if nb_bestidx == (NUM_NARROWBANDS - 1) / 2:
                    bars[nb_bestidx].set_facecolor('g')
                else:
                    bars[nb_bestidx].set_facecolor('r')
                # the RMS power of the data
                rms = numpy.sqrt(sum(cum ** 2) / float(len(cum)))
                pow_axes = plt.subplot(312)
                pow_axes.axis('off')
                plt.xlim(xmin=0, xmax=RMS_RANGES[-1][-1])
                plt.ylim(ymin=0, ymax=1)
                pow_axes.text(0.5, 0.5,
                              'Main Freq: {0:.1f} Hz\n'
                              'String: {1}\n'
                              'RMS Power: {2:.1f}'.format(
                        main_freq, string, rms))
                color = 'g'
                for (rms_color, rms_min, _rms_max) in RMS_RANGES:
                    if rms_min <= rms:
                        color = rms_color
                plt.barh(bottom=0.2, height=0.2, width=rms, color=color)
                # draw the current frequency spectrum
                freq_axes = plt.subplot(313)
                plt.title('Frequency Spectrum')
                plt.xlabel('Frequency (Hz)')
                plt.xlim(xmin=0, xmax=3000) # confine to 3 kHz
                freq_axes.get_yaxis().set_ticks([]) # remove y ticks
                plt.plot(freqs, cum)
                plt.draw()
                # reset accumulators
                cum = numpy.zeros(len(freqs))
                frame = numpy.array([], dtype=numpy.int16)
            i += 1
    except KeyboardInterrupt:
        pass

    print
    print "* done recording"

    stream.close()
    audio.terminate()

if __name__ == '__main__' and sys.argv != ['']:
    main()
