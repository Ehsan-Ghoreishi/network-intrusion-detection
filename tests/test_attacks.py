import pytest

from nids.attacks import ATTACK_CATEGORIES, to_category
from nids.config import CLASSES


def test_normal_maps_to_normal():
    assert to_category("normal") == "normal"


@pytest.mark.parametrize("label, family", [
    ("neptune", "dos"), ("mailbomb", "dos"), ("portsweep", "probe"),
    ("saint", "probe"), ("guess_passwd", "r2l"), ("snmpguess", "r2l"),
    ("buffer_overflow", "u2r"), ("sqlattack", "u2r"),
])
def test_known_labels(label, family):
    assert to_category(label) == family


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="Unknown NSL-KDD label"):
        to_category("not_an_attack")


def test_mapping_covers_all_39_attack_types():
    assert len(ATTACK_CATEGORIES) == 39


def test_families_are_known_classes():
    assert set(ATTACK_CATEGORIES.values()) == set(CLASSES) - {"normal"}
