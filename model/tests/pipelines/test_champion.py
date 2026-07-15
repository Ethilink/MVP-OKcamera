import numpy as np

from orc_model.pipelines.matching.champion import ChampionGallery, ChampionMethod
from orc_model.pipelines.matching.interface import REJECT


def matcher_with_vector_inputs(*, cos_tau: float = 0.6) -> ChampionMethod:
    """Build the numerical matcher without loading DINO for focused score tests."""
    matcher = object.__new__(ChampionMethod)
    matcher.alpha = 0.0003
    matcher.tau = 0.30
    matcher.margin = 0.02
    matcher.cos_tau = cos_tau
    matcher.max_iter = 2000
    matcher._embed_fused = lambda crops, _masks: np.asarray(crops, dtype=np.float64)
    return matcher


def test_single_gallery_cosine_gate_accepts_matching_vector():
    matcher = matcher_with_vector_inputs()
    gallery = ChampionGallery(views=np.array([[1.0, 0.0]]), identity=1)

    scores = matcher.score([[1.0, 0.0]], [None], {}, {1: gallery})

    assert matcher.accept(scores) == 1


def test_single_gallery_cosine_gate_rejects_orthogonal_vector():
    matcher = matcher_with_vector_inputs()
    gallery = ChampionGallery(views=np.array([[1.0, 0.0]]), identity=1)

    scores = matcher.score([[0.0, 1.0]], [None], {}, {1: gallery})

    assert scores[1] == 0.0
    assert matcher.accept(scores) == REJECT
