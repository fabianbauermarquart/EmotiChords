# EmotiChords

*EmotiChords* uses machine learning to analyze the emotion of a written piece of text and plays back that emotion as a
musical chord.

## Description

An bidirectional LSTM model trained on Google's
[GoEmotions dataset](https://www.tensorflow.org/datasets/catalog/goemotions)
extracts the emotion of the input text.

Then, a rule-based algorithm retrieves the emotion's respective musical chords according to
[this chart](https://preview.redd.it/7f94b4mkuyy21.jpg?width=4398&format=pjpg&auto=webp&s=39a10b4467f605ca01f2c3f3e82c8923a95f1baf)
. The final chords are synthesized using digital signal processing techniques.

## Usage

    main.py -t 'This is great news!'

### Example Chords

- annoyance: M9, M7b5
- fear: M7b5, m
- joy: M, M9
- love: Maj7, add9
- nervousness: sus2, dim
- optimism: Maj7, M6
- sadness: add9, madd9

## Requirements

- tfds-nightly
- tensorflow
- keras
- numpy
- scipy
- pygame
- librosa


