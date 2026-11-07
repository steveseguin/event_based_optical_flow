import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.event_image_converter import EventImageConverter


@pytest.mark.parametrize("method", ["bilinear_vote", "count"])
@pytest.mark.parametrize("sigma", [0, 0.5, 1, 2])
def test_numpy_blur_keeps_batches_independent(method, sigma):
    imager = EventImageConverter((9, 11))
    events = np.array(
        [
            [[2.25, 3.5, 0.0, 1], [3.0, 4.0, 0.1, 0]],
            [[6.0, 7.5, 0.0, 0], [5.5, 8.0, 0.1, 1]],
            [[1.0, 8.0, 0.0, 1], [7.0, 1.0, 0.1, 0]],
        ]
    )
    weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    actual = imager.create_image_from_events_numpy(events, method, weights, sigma)
    expected = np.stack(
        [
            imager.create_image_from_events_numpy(batch, method, weight, sigma)
            for batch, weight in zip(events, weights)
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_numpy_blur_does_not_add_events_to_an_empty_batch():
    imager = EventImageConverter((9, 11))
    # A warped event outside the image contributes nothing to the second batch.
    events = np.array([[[4.0, 5.0, 0.0, 1]], [[-100.0, -100.0, 0.0, 1]]])

    image = imager.create_iwe(events, sigma=1)

    np.testing.assert_allclose(image[0].sum(), 1.0)
    np.testing.assert_array_equal(image[1], np.zeros((9, 11)))


@pytest.mark.parametrize("sigma", [0, 0.5, 1, 2])
def test_numpy_blur_keeps_polarities_independent(sigma):
    imager = EventImageConverter((9, 11))
    events = np.array([[2.25, 3.5, 0.0, 1], [6.0, 7.5, 0.1, 0]])
    weights = np.array([2.0, 5.0])
    unblurred = imager.create_image_from_events_numpy(events, "polarity", weights, sigma=0)

    actual = imager.create_image_from_events_numpy(events, "polarity", weights, sigma)
    expected = np.stack([gaussian_filter(channel, sigma) for channel in unblurred])

    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(actual.sum(axis=(-2, -1)), [2.0, 5.0])


def test_numpy_blur_does_not_invent_opposite_polarity():
    imager = EventImageConverter((9, 11))
    events = np.array([[4.0, 5.0, 0.0, 1]])

    image = imager.create_image_from_events_numpy(events, "polarity", sigma=1)

    np.testing.assert_allclose(image[0].sum(), 1.0)
    np.testing.assert_array_equal(image[1], np.zeros((9, 11)))


@pytest.mark.parametrize("method", ["bilinear_vote", "count"])
@pytest.mark.parametrize("sigma", [0.5, 1, 2])
def test_numpy_single_image_blur_is_unchanged(method, sigma):
    imager = EventImageConverter((9, 11))
    events = np.array([[2.25, 3.5, 0.0, 1], [6.0, 7.5, 0.1, 0]])
    unblurred = imager.create_image_from_events_numpy(events, method, sigma=0)

    actual = imager.create_image_from_events_numpy(events, method, sigma=sigma)

    np.testing.assert_array_equal(actual, gaussian_filter(unblurred, sigma))
