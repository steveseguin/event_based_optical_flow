import numpy as np
import pytest
import torch

from src import utils


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_identical_flows_have_finite_angular_error(dtype, backend):
    # Some of these finite vectors round the computed cosine above 1.
    values = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.1, 2.0, 3.0, 5.0, 10.0, 20.0])
    x, y = np.meshgrid(values, values)
    flow = np.stack([x, y], axis=0)[None].astype(dtype)

    if backend == "numpy":
        error = utils.calculate_flow_error_numpy(flow, flow)
    else:
        tensor = torch.from_numpy(flow)
        error = utils.calculate_flow_error_tensor(tensor, tensor)

    assert np.isfinite(float(error["AE"]))
    assert float(error["AE"]) == pytest.approx(0.0, abs=1e-3)
    assert float(error["EPE"]) == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_angular_error_preserves_valid_cosines(dtype, backend):
    # Homogeneous flow vectors (1, 1, 1) and (1, -2, 1) are orthogonal.
    predicted = np.array([[[[1.0]], [[1.0]]]], dtype=dtype)
    actual = np.array([[[[1.0]], [[-2.0]]]], dtype=dtype)

    if backend == "numpy":
        error = utils.calculate_flow_error_numpy(actual, predicted)
    else:
        error = utils.calculate_flow_error_tensor(
            torch.from_numpy(actual), torch.from_numpy(predicted)
        )

    # The metric keeps an existing 1e-5 stabilizer in its valid-point count.
    assert float(error["AE"]) == pytest.approx((np.pi / 2) / (1 + 1e-5), rel=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_angular_error_stays_finite_with_masked_pixels(backend):
    flow = np.array([[[[0.1, 1.1]], [[0.3, 0.7]]]], dtype=np.float32)
    mask = np.array([[[[True, False]]]])

    if backend == "numpy":
        error = utils.calculate_flow_error_numpy(flow, flow, mask)
    else:
        tensor = torch.from_numpy(flow)
        error = utils.calculate_flow_error_tensor(tensor, tensor, torch.from_numpy(mask))

    assert np.isfinite(float(error["AE"]))
    assert float(error["AE"]) == pytest.approx(0.0, abs=1e-3)
