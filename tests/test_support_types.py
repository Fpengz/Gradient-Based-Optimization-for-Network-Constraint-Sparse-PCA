import numpy as np

from topospca.synthetic.support import generate_supports


def test_multi_cluster_support_sizes():
    rng = np.random.default_rng(0)
    p = 20
    adjacency = np.zeros((p, p))
    for i in range(p - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    supports = generate_supports(
        p=p,
        r=1,
        support_size=6,
        support_type="multi_cluster",
        rng=rng,
        adjacency=adjacency,
    )
    assert len(supports) == 1
    assert len(supports[0]) == 6


def test_fragmented_support_sizes():
    rng = np.random.default_rng(1)
    p = 15
    adjacency = np.zeros((p, p))
    for i in range(p - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    supports = generate_supports(
        p=p,
        r=1,
        support_size=5,
        support_type="fragmented",
        rng=rng,
        adjacency=adjacency,
    )
    assert len(supports[0]) == 5


def test_cross_community_requires_labels():
    rng = np.random.default_rng(2)
    p = 10
    try:
        generate_supports(
            p=p,
            r=1,
            support_size=4,
            support_type="cross_community",
            rng=rng,
            adjacency=np.zeros((p, p)),
            metadata={},
        )
    except ValueError as exc:
        assert "sbm_labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing sbm_labels")


def test_cross_community_support_size():
    rng = np.random.default_rng(3)
    p = 12
    labels = np.array([0] * 6 + [1] * 6)
    supports = generate_supports(
        p=p,
        r=1,
        support_size=6,
        support_type="cross_community",
        rng=rng,
        adjacency=np.zeros((p, p)),
        metadata={"sbm_labels": labels},
    )
    assert len(supports[0]) == 6
