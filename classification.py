import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

LABELS = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral',
]


def build_dataset(eval=False, inspect=False):
    if not eval:
        split = 'train'
    else:
        split = 'test'

    batch_size = 1024

    if inspect:
        batch_size = 1

        # Convert examples from their dataset format into the model format.

    def process_input(features):
        # Generate the projection for each comment_text input.  The final tensor
        # will have the shape [batch_size, number of tokens, feature size].
        # Additionally, we generate a tensor containing the number of tokens for
        # each comment_text (seq_length).  This is needed because the projection
        # tensor is a full tensor, and we are not using EOS tokens.
        text = features['comment_text']
        text = tf.reshape(text, [batch_size])

        # Convert the labels into an indicator tensor, using the LABELS indices.
        label = tf.stack([features[label] for label in LABELS], axis=-1)
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [batch_size, len(LABELS)])

        model_features = ({'text': text}, label)

        if inspect:
            model_features = (model_features[0], model_features[1], features)

        return model_features

    ds = tfds.load('goemotions', split=split)
    ds = ds.shuffle(buffer_size=batch_size * 2)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(process_input)
    # ds = ds.map(lambda y: tf.py_function((lambda x: process_input(x)), inp=[y], Tout=(tf.bool, tf.string)))
    return ds


def get_text_only(x, _):
    return x['text']


def create_model():
    train_ds = build_dataset(eval=False)
    test_ds = build_dataset(eval=True)
    text_only_train_ds = train_ds.map(get_text_only)

    max_length = 600
    max_tokens = 20000
    text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_length,
    )

    text_vectorization.adapt(text_only_train_ds)

    int_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x['text']), y),
        num_parallel_calls=4)
    int_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x['text']), y),
        num_parallel_calls=4)

    inputs = tf.keras.Input(shape=(None,), dtype="int64")
    embedded = layers.Embedding(
        input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(embedded)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(LABELS), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras",
                                           save_best_only=True)
    ]

    model.fit(int_train_ds, validation_data=int_test_ds, epochs=40, callbacks=callbacks)
    model = tf.keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
    print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


if __name__ == '__main__':
    create_model()
