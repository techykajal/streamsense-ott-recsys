"""
trainer_module.py — module_file shared by TFX Transform + Trainer.

preprocessing_fn : feature engineering inside the Transform graph.
run_fn           : trains a small Keras ranking model (predict P(engagement)).
This is the TF-side ranker; the PyTorch ranker (ranking_torch.py) is the Triton/TorchServe one.
"""
import tensorflow as tf
import tensorflow_transform as tft

USER, MOVIE, SEG, LABEL = "user_id", "movie_id", "segment", "label"


def preprocessing_fn(inputs):
    out = {}
    out[USER] = tft.compute_and_apply_vocabulary(inputs[USER], top_k=50_000, vocab_filename="user_vocab")
    out[MOVIE] = tft.compute_and_apply_vocabulary(inputs[MOVIE], top_k=50_000, vocab_filename="movie_vocab")
    out[SEG] = tf.cast(inputs[SEG], tf.int64)
    out[LABEL] = tf.cast(inputs[LABEL], tf.float32)
    return out


def _input_fn(files, tf_transform_output, batch=1024):
    spec = tf_transform_output.transformed_feature_spec().copy()
    ds = tf.data.experimental.make_batched_features_dataset(
        files, batch, spec, label_key=LABEL, shuffle=True, num_epochs=None)
    return ds


def _model(n_user, n_movie, n_seg, dim=32):
    u = tf.keras.Input(shape=(1,), name=USER, dtype=tf.int64)
    m = tf.keras.Input(shape=(1,), name=MOVIE, dtype=tf.int64)
    s = tf.keras.Input(shape=(1,), name=SEG, dtype=tf.int64)
    ue = tf.keras.layers.Embedding(n_user + 1, dim)(u)
    me = tf.keras.layers.Embedding(n_movie + 1, dim)(m)
    se = tf.keras.layers.Embedding(n_seg + 1, dim)(s)
    x = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(t) for t in (ue, me, se)])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model([u, m, s], out)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def run_fn(fn_args):
    tfto = tft.TFTransformOutput(fn_args.transform_output)
    train = _input_fn(fn_args.train_files, tfto)
    val = _input_fn(fn_args.eval_files, tfto)
    n_user = tfto.vocabulary_size_by_name("user_vocab")
    n_movie = tfto.vocabulary_size_by_name("movie_vocab")
    model = _model(n_user, n_movie, n_seg=64)
    model.fit(train, steps_per_epoch=fn_args.train_steps,
              validation_data=val, validation_steps=fn_args.eval_steps, epochs=1)
    model.save(fn_args.serving_model_dir, save_format="tf")
