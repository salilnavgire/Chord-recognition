Chord-recognition
=================

Project is very unorganized right now because it was one of my class assignments. Will be cleaning it up soon.

Chord recognition in songs using Viterbi algorithm - Wrote a Python code which takes any audio file and converts it into a chromogram.  It then computes the distance matrix between features and template, further applying Viterbi algorithm to generate corresponding chords.

First run fitness_matrix.py to generate a fitness matrix for a song
Then run chord_meanfilter.py for mean filter chord recognition or run chord_viterbi.py for viterbi algorithm chord recognition.
