import numpy as np

from nids.models import build_autoencoder, build_classifier, reconstruction_error
from nids.supervised import balanced_class_weights
from nids.utils import set_seed

RNG = np.random.default_rng(0)


def test_classifier_outputs_probabilities():
    set_seed(0)
    model = build_classifier(20, 5)
    proba = model.predict(RNG.random((8, 20), dtype="float32"), verbose=0)
    assert proba.shape == (8, 5)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


def test_autoencoder_reconstructs_input_shape():
    model = build_autoencoder(20, bottleneck=4)
    assert model.output_shape == (None, 20)
    assert model.get_layer("bottleneck").output.shape[-1] == 4


def test_reconstruction_error_is_per_row():
    model = build_autoencoder(10)
    X = RNG.random((16, 10), dtype="float32")
    err = reconstruction_error(model, X)
    assert err.shape == (16,) and (err >= 0).all()


def test_balanced_weights_favour_rare_classes():
    y = np.array([0] * 90 + [1] * 9 + [2])
    w = balanced_class_weights(y)
    assert w[2] > w[1] > w[0]


def test_classifier_learns_separable_data():
    set_seed(0)
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-2, 1, (200, 4)), rng.normal(2, 1, (200, 4))].astype("float32")
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    model = build_classifier(4, 2)
    model.fit(X, y, epochs=15, batch_size=32, verbose=0)
    assert (model.predict(X, verbose=0).argmax(axis=1) == y).mean() > 0.95
