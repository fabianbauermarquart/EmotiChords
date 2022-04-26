import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

import argparse

from chord_generation import play_progression
from classification import create_model, build_dataset, get_text_only, LABELS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmotiChords')
    parser.add_argument('-t', '--text', help='Input text', required=True)

    args = parser.parse_args()

    try:
        model = tf.keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
    except OSError:
        create_model()
        model = tf.keras.models.load_model("embeddings_bidir_gru_with_masking.keras")

    train_ds = build_dataset(eval=False)
    text_only_train_ds = train_ds.map(get_text_only)

    max_length = 600
    max_tokens = 20000
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_length,
    )

    text_vectorization.adapt(text_only_train_ds)

    label_index = np.argmax(model.call(text_vectorization((args.text,))))
    label = LABELS[label_index]

    play_progression(label)
