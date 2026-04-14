import numpy as np
from norfair_rs import Detection, Tracker


def test_reid_with_embeddings():
    """
    Test that ReID works correctly using embeddings even when positions are far apart.
    """

    # 1. Setup tracker with a ReID distance function that uses embeddings
    def reid_dist(obj1, obj2):
        # obj1 is the "candidate" (temporary Detection created from TrackedObject)
        # obj2 is the "old object" (TrackedObject)
        emb1 = obj1.embedding
        emb2 = obj2.last_detection.embedding

        if emb1 is None or emb2 is None:
            return 1000.0

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1.0 - similarity

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=1.0,
        initialization_delay=1,
        hit_counter_max=5,
        reid_distance_function=reid_dist,
        reid_distance_threshold=0.1,
        reid_hit_counter_max=5,
    )

    # 2. Create an object with an embedding
    emb_a = np.array([1.0, 0.0, 0.0])
    det1 = Detection(np.array([[10.0, 10.0]]), embedding=emb_a)
    tracker.update([det1])  # initializing
    tracked = tracker.update([det1])  # permanent
    assert len(tracked) == 1
    obj_id = tracked[0].id

    # 3. Let the object die
    for _ in range(7):
        tracker.update([])
    assert len(tracker.get_active_objects()) == 0

    # 4. Trigger ReID
    det2 = Detection(np.array([[100.0, 100.0]]), embedding=emb_a)

    # Frame A: Create NEW initializing object for det2
    tracker.update([det2])

    # Frame B: Match that initializing object to det2 again
    # This makes it a "matched_init_obj" which is a candidate for ReID
    tracker.update([det2])

    # Check ReID: The old ID should have been restored
    all_objs = tracker.tracked_objects
    matched = [o for o in all_objs if np.allclose(o.estimate, [[100.0, 100.0]])]
    assert len(matched) == 1
    assert matched[0].id == obj_id


def test_reid_callback_types():
    """
    Verify the types passed to the reid_distance_function callback.
    """
    types_seen = []

    def reid_dist(obj1, obj2):
        types_seen.append((type(obj1).__name__, type(obj2).__name__))
        if hasattr(obj1, "embedding") and obj1.embedding is not None:
            types_seen.append("embedding_found")
        return 0.0

    tracker = Tracker(
        distance_function="euclidean",
        distance_threshold=1.0,
        initialization_delay=1,
        hit_counter_max=5,
        reid_distance_function=reid_dist,
        reid_distance_threshold=1.0,
        reid_hit_counter_max=5,
    )

    # Setup an object
    tracker.update([Detection(np.array([[10.0, 10.0]]), embedding=np.array([1.0]))])
    tracker.update([Detection(np.array([[10.0, 10.0]]), embedding=np.array([1.0]))])
    for _ in range(7):
        tracker.update([])

    # Trigger ReID
    det2 = Detection(np.array([[100.0, 100.0]]), embedding=np.array([1.0]))
    tracker.update([det2])  # Create init obj
    tracker.update([det2])  # Match init obj -> Trigger ReID

    assert len(types_seen) > 0
    assert types_seen[0] == ("Detection", "TrackedObject")
    assert "embedding_found" in types_seen
