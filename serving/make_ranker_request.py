"""
make_ranker_request.py — build a TF-Serving predict request for the TFX ranker.

The ranker's serving_default signature parses a serialized tf.Example (input "examples")
and returns "prediction". The tf.Example must carry the raw features the schema expects
(everything except the label). `movie_title` and `rating` are not used by the model's
transform, but must be PRESENT for tf.io.parse_example — so we send neutral placeholders.

    python serving/make_ranker_request.py <user_id> <movie_id> <segment>  ->  prints request JSON
"""
import sys, json, base64
import tensorflow as tf


def _b(v):  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(v).encode()]))
def _i(v):  return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))
def _f(v):  return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))


def serialized_example(user_id, movie_id, segment, movie_title="", rating=0.0):
    ex = tf.train.Example(features=tf.train.Features(feature={
        "user_id":     _b(user_id),
        "movie_id":    _b(movie_id),
        "movie_title": _b(movie_title),   # present but unused by the model
        "segment":     _i(segment),
        "rating":      _f(rating),        # present but unused by the model
    }))
    return ex.SerializeToString()


def request_json(triples):
    """triples: list of (user_id, movie_id, segment[, movie_title]) -> TF Serving 'inputs' JSON."""
    examples = []
    for t in triples:
        title = t[3] if len(t) > 3 else ""
        examples.append({"b64": base64.b64encode(
            serialized_example(t[0], t[1], t[2], title)).decode()})
    return {"signature_name": "serving_default", "inputs": {"examples": examples}}


if __name__ == "__main__":
    u, m, s = sys.argv[1], sys.argv[2], sys.argv[3]
    print(json.dumps(request_json([(u, m, s)])))
