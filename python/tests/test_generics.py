"""
Runtime support for the generic type parameter.

`Detection`, `TrackedObject` and `Tracker` are generic in the type stubs,
parameterized by the type of the user payload carried in `Detection.data`.
PyO3 classes are not subscriptable by default, so the classes implement
`__class_getitem__` to keep `Detection[MyPayload]` valid at runtime too.

Payload round-tripping itself is covered by `test_compatibility.py`.
"""

import pytest
from norfair_rs import Detection, TrackedObject, Tracker


@pytest.mark.parametrize("cls", [Detection, TrackedObject, Tracker])
def test_class_is_subscriptable(cls):
    alias = cls[dict]
    assert alias.__origin__ is cls
    assert alias.__args__ == (dict,)
