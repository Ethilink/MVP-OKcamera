import numpy as np
import pytest

from orc_model.components.detector._rfdetr_postprocess import decode_predictions, preprocess

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.mark.parametrize("height,width", [(1080, 1920), (480, 480)])
def test_output_shape_is_fixed_regardless_of_input_size(height, width):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    output = preprocess(image)

    assert output.shape == (1, 3, 768, 768)


def test_output_dtype_is_float32():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    output = preprocess(image)

    assert output.dtype == np.float32


def test_normalization_matches_hand_computed_imagenet_formula():
    # Real-world color: R=200, G=100, B=50. Stored BGR (cv2 convention), so the
    # array's channel order is [B, G, R] = [50, 100, 200].
    r, g, b = 200, 100, 50
    image = np.empty((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = b
    image[:, :, 1] = g
    image[:, :, 2] = r

    output = preprocess(image)

    expected_r = (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_b = (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    # Output is NCHW with RGB channel order: channel 0 = R, 1 = G, 2 = B.
    assert np.allclose(output[0, 0], expected_r, atol=1e-6)
    assert np.allclose(output[0, 1], expected_g, atol=1e-6)
    assert np.allclose(output[0, 2], expected_b, atol=1e-6)


def test_bgr_to_rgb_channel_swap_is_detectable():
    # Distinct, non-symmetric values per channel so a BGR/RGB mixup produces a
    # detectably wrong (not accidentally correct) result.
    r, g, b = 10, 150, 240
    image = np.empty((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = b
    image[:, :, 1] = g
    image[:, :, 2] = r

    output = preprocess(image)

    expected_r = (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_b = (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]

    assert np.allclose(output[0, 0], expected_r, atol=1e-6)
    assert np.allclose(output[0, 1], expected_g, atol=1e-6)
    assert np.allclose(output[0, 2], expected_b, atol=1e-6)

    # Sanity: channel 0 of the output should NOT match the raw B value's
    # expected-as-R normalization unless B == R (it doesn't here), confirming
    # the swap actually happened rather than being a no-op.
    wrong_r_from_b = (b / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert not np.allclose(output[0, 0], wrong_r_from_b)


# --- decode_predictions ------------------------------------------------------

_LOW_LOGIT = -10.0  # sigmoid(-10) ~= 4.5e-5 -- effectively "never a real detection"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _make_inputs(num_queries: int, num_classes: int = 2, mask_size: int = 20):
    """All-low-score synthetic (dets, labels, masks) with batch dim 1, ready
    for individual queries/classes to be overridden per test. `mask_size` is
    deliberately small (not the real model's 192) -- decode_predictions reads
    the mask's spatial size off the array itself, so tests don't need to pay
    for full-size arrays."""
    dets = np.zeros((1, num_queries, 4), dtype=np.float32)
    labels = np.full((1, num_queries, num_classes), _LOW_LOGIT, dtype=np.float32)
    masks = np.full((1, num_queries, mask_size, mask_size), _LOW_LOGIT, dtype=np.float32)
    return dets, labels, masks


def test_channel_filtering_excludes_untrained_channel_even_at_higher_score():
    dets, labels, masks = _make_inputs(num_queries=2)

    # Query 0: real channel (class_idx=0), moderate-high score.
    labels[0, 0, 0] = 2.0  # sigmoid ~ 0.881
    dets[0, 0] = [0.5, 0.5, 0.2, 0.2]

    # Query 1: real channel scores LOWER than query 0 but still above
    # threshold; its untrained channel (class_idx=1) scores HIGHER than every
    # real-channel row in this input.
    labels[0, 1, 0] = 0.5  # sigmoid ~ 0.622 (lower than query 0, still >= 0.5)
    labels[0, 1, 1] = 10.0  # sigmoid ~ 0.99995 (highest score overall, wrong channel)
    dets[0, 1] = [0.3, 0.3, 0.1, 0.1]

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=100,
        image_height=100,
        confidence_threshold=0.5,
        top_k=10,
    )

    assert len(result) == 2
    # The untrained channel's score must never appear, however high it is.
    assert not np.any(np.isclose(result.confidence, _sigmoid(10.0), atol=1e-4))
    # The lower-scoring but real-channel row from query 1 IS included --
    # proves filtering is by class_idx, not just by score.
    assert np.any(np.isclose(result.confidence, _sigmoid(0.5), atol=1e-4))
    assert np.all(result.class_id == 0)
    assert list(result.data["class_name"]) == ["surgical_instrument"] * 2


def test_untrained_channel_cannot_consume_the_real_channel_top_k_budget():
    dets, labels, masks = _make_inputs(num_queries=300)
    labels[0, :, 1] = 5.0  # all 300 untrained logits outrank the real detection
    labels[0, 0, 0] = 4.0
    dets[0, 0] = [0.5, 0.5, 0.2, 0.2]

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=100,
        image_height=100,
        confidence_threshold=0.5,
        top_k=300,
    )

    assert len(result) == 1
    assert np.isclose(result.confidence[0], _sigmoid(4.0))


def test_shared_index_gather_keeps_box_and_mask_aligned_to_same_query():
    num_queries = 5
    mask_size = 20
    dets, labels, masks = _make_inputs(num_queries=num_queries, mask_size=mask_size)

    # Distinguishable box per query.
    for i in range(num_queries):
        dets[0, i] = [0.1 * (i + 1), 0.1 * (i + 1), 0.05, 0.05]

    # Distinguishable mask "signature": query i gets +10 in its own unique
    # horizontal strip of rows, -10 (the default) everywhere else.
    strip_rows = mask_size // num_queries  # 4
    for i in range(num_queries):
        masks[0, i, i * strip_rows : (i + 1) * strip_rows, :] = 10.0

    # Only query 2's real channel clears the threshold; every other
    # query/channel stays at the low default.
    survivor = 2
    labels[0, survivor, 0] = 5.0  # sigmoid ~ 0.993

    image_width, image_height = 200, 100  # non-square, catches W/H mixups

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=0.5,
        top_k=10,
    )

    assert len(result) == 1

    cx, cy, w, h = dets[0, survivor]
    expected_xyxy = np.array(
        [
            (cx - w / 2) * image_width,
            (cy - h / 2) * image_height,
            (cx + w / 2) * image_width,
            (cy + h / 2) * image_height,
        ]
    )
    assert np.allclose(result.xyxy[0], expected_xyxy)

    # The mask's True region must line up with query `survivor`'s own strip
    # (rescaled from mask_size to image_height), not any other query's strip
    # -- this is what would fail if the box gather and mask gather ever used
    # different indices.
    row_scale = image_height / mask_size
    survivor_row_start = int(survivor * strip_rows * row_scale)
    survivor_row_end = int((survivor + 1) * strip_rows * row_scale)
    survivor_interior_row = (survivor_row_start + survivor_row_end) // 2
    interior_col = image_width // 2
    assert result.mask[0, survivor_interior_row, interior_col]

    other = 0
    other_row_start = int(other * strip_rows * row_scale)
    other_row_end = int((other + 1) * strip_rows * row_scale)
    other_interior_row = (other_row_start + other_row_end) // 2
    assert not result.mask[0, other_interior_row, interior_col]


def test_box_coordinates_match_hand_computed_cxcywh_to_xyxy_rescale():
    dets, labels, masks = _make_inputs(num_queries=1)
    dets[0, 0] = [0.4, 0.6, 0.2, 0.1]  # cx, cy, w, h -- normalized to 768x768 input
    labels[0, 0, 0] = 5.0  # well above threshold

    image_width, image_height = 1920, 1080  # non-square, catches a W/H swap

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=0.5,
        top_k=10,
    )

    assert len(result) == 1
    # Hand-computed: x1=(0.4-0.1)*1920=576, y1=(0.6-0.05)*1080=594,
    # x2=(0.4+0.1)*1920=960, y2=(0.6+0.05)*1080=702.
    expected = np.array([576.0, 594.0, 960.0, 702.0])
    assert np.allclose(result.xyxy[0], expected)


def test_mask_resize_and_threshold_preserve_sign_pattern_on_interior_points():
    dets, labels, masks = _make_inputs(num_queries=1, mask_size=20)
    labels[0, 0, 0] = 5.0

    # Top half of the 20x20 mask logits positive, bottom half negative.
    masks[0, 0, :10, :] = 10.0
    masks[0, 0, 10:, :] = -10.0

    image_width, image_height = 300, 100  # non-square target

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=0.5,
        top_k=10,
    )

    assert result.mask.shape == (1, image_height, image_width)
    # Interior points well away from the resized boundary at row 50.
    assert result.mask[0, 20, 150]
    assert not result.mask[0, 80, 150]


def test_confidence_threshold_boundary_is_inclusive():
    dets, labels, masks = _make_inputs(num_queries=3)
    # sigmoid(0.0) == 0.5 exactly, so this hits the >= threshold precisely.
    labels[0, 0, 0] = 0.0  # exactly at threshold
    labels[0, 1, 0] = 0.1  # just above threshold
    labels[0, 2, 0] = -0.1  # just below threshold

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=100,
        image_height=100,
        confidence_threshold=0.5,
        top_k=10,
    )

    assert len(result) == 2
    assert np.any(np.isclose(result.confidence, _sigmoid(0.0)))
    assert np.any(np.isclose(result.confidence, _sigmoid(0.1)))
    assert not np.any(np.isclose(result.confidence, _sigmoid(-0.1)))


def test_top_k_truncates_to_highest_scoring_survivors():
    dets, labels, masks = _make_inputs(num_queries=5)
    score_logits = [1.0, 2.0, 3.0, 4.0, 5.0]  # all class_idx=0, all above threshold
    for i, logit in enumerate(score_logits):
        labels[0, i, 0] = logit

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=100,
        image_height=100,
        confidence_threshold=0.5,
        top_k=3,
    )

    assert len(result) == 3
    expected_top3 = sorted((_sigmoid(logit) for logit in score_logits), reverse=True)[:3]
    assert np.allclose(sorted(result.confidence, reverse=True), expected_top3)


def test_zero_survivors_returns_valid_empty_detections():
    dets, labels, masks = _make_inputs(num_queries=4)
    # All real-channel scores stay at the low default, well below threshold.

    result = decode_predictions(
        dets,
        labels,
        masks,
        image_width=64,
        image_height=48,
        confidence_threshold=0.5,
        top_k=10,
    )

    assert len(result) == 0
    assert result.xyxy.shape == (0, 4)
    assert result.mask.shape == (0, 48, 64)
    assert result.confidence.shape == (0,)
    assert result.class_id.shape == (0,)
    assert result.data["class_name"].shape == (0,)
