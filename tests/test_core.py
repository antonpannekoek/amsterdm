import numpy as np
from numpy.testing import assert_allclose
import pytest

from amsterdm.core import downsample


@pytest.fixture
def base_1d():
    return np.arange(1, 9, dtype=float)


@pytest.fixture
def data_2d(base_1d):
    data = base_1d.reshape(8, 1)
    data = np.repeat(data, 3, axis=1)
    assert data.shape == (8, 3)
    return data


@pytest.fixture
def data_3d(base_1d):
    data = base_1d.reshape(8, 1, 1)
    data = np.repeat(data, 3, axis=1)
    data = np.repeat(data, 4, axis=2)
    assert data.shape == (8, 3, 4)
    return data


@pytest.fixture
def data_4d(base_1d):
    data = base_1d.reshape(8, 1, 1, 1)
    data = np.repeat(data, 3, axis=1)
    data = np.repeat(data, 4, axis=2)
    data = np.repeat(data, 5, axis=3)
    assert data.shape == (8, 3, 4, 5)
    return data


@pytest.fixture(params=[2, 3, 4])  # , name="nddata")
def nddata(base_1d, request):
    if request.param == 2:
        nddata = base_1d.reshape(8, 1)
        nddata = np.repeat(nddata, 3, axis=1)
        assert nddata.shape == (8, 3)
    elif request.param == 3:
        nddata = base_1d.reshape(8, 1, 1)
        nddata = np.repeat(nddata, 3, axis=1)
        nddata = np.repeat(nddata, 4, axis=2)
        assert nddata.shape == (8, 3, 4)
    elif request.param == 4:
        nddata = base_1d.reshape(8, 1, 1, 1)
        nddata = np.repeat(nddata, 3, axis=1)
        nddata = np.repeat(nddata, 4, axis=2)
        nddata = np.repeat(nddata, 5, axis=3)
        assert nddata.shape == (8, 3, 4, 5)

    return nddata


@pytest.mark.parametrize(
    "remainder, factor, method, means",
    [
        ("droptail", 1, "mean", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("droptail", 2, "mean", [1.5, 3.5, 5.5, 7.5]),
        ("droptail", 4, "mean", [2.5, 6.5]),
        # not integer division of nbins
        ("droptail", 3, "mean", [2, 5]),
        ("droptail", 5, "mean", [3]),
        ("droptail", 9, "mean", [4.5]),
        ("droptail", 1, "sum", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("droptail", 4, "sum", [10, 26]),
        ("droptail", 3, "sum", [6, 15]),
        ("droptail", 5, "sum", [15]),
        ("droptail", 9, "sum", [36]),
        ("addtail", 1, "mean", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("addtail", 2, "mean", [1.5, 3.5, 5.5, 7.5]),
        ("addtail", 4, "mean", [2.5, 6.5]),
        # not integer division of bins
        ("addtail", 3, "mean", [2, 6]),
        ("addtail", 5, "mean", [4.5]),
        ("addtail", 9, "mean", [4.5]),
        ("addtail", 1, "sum", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("addtail", 4, "sum", [10, 26]),
        ("addtail", 3, "sum", [6, 30]),
        ("addtail", 5, "sum", [36]),
        ("addtail", 9, "sum", [36]),
        ("drophead", 1, "mean", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("drophead", 2, "mean", [1.5, 3.5, 5.5, 7.5]),
        ("drophead", 4, "mean", [2.5, 6.5]),
        # not integer division of nbins
        ("drophead", 3, "mean", [4, 7]),
        ("drophead", 5, "mean", [6]),
        ("drophead", 9, "mean", [4.5]),
        ("drophead", 1, "sum", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("drophead", 4, "sum", [10, 26]),
        ("drophead", 3, "sum", [12, 21]),
        ("drophead", 5, "sum", [30]),
        ("drophead", 9, "sum", [36]),
        ("addhead", 1, "mean", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("addhead", 2, "mean", [1.5, 3.5, 5.5, 7.5]),
        ("addhead", 4, "mean", [2.5, 6.5]),
        # not integer division of nbins
        ("addhead", 3, "mean", [3, 7]),
        ("addhead", 5, "mean", [4.5]),
        ("drophead", 9, "mean", [4.5]),
        ("addhead", 1, "sum", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("addhead", 4, "sum", [10, 26]),
        ("addhead", 3, "sum", [15, 21]),
        ("addhead", 5, "sum", [36]),
        ("addhead", 9, "sum", [36]),
    ],
)
class TestDownsample:
    def test_downsample_nd(self, nddata, remainder, factor, method, means):
        result = downsample(nddata, factor=factor, remainder=remainder, method=method)

        # Build expected array from the list of means
        expected = np.asarray(
            [m * np.ones(nddata.shape[1:], dtype=float) for m in means]
        )
        assert_allclose(result, expected, strict=True)


def test_downsample_errors():
    data = np.arange(1, 9, dtype=float)

    with pytest.raises(ValueError, match="'data' should at least be two-dimensional"):
        downsample(data, factor=2)

    data = data.reshape(8, 1)
    data = np.repeat(data, 3, axis=1)

    with pytest.raises(ValueError, match="'factor' should be a positive integer"):
        downsample(data, factor=0)

    with pytest.raises(ValueError, match="'method' should be one of 'mean' or 'sum'"):
        downsample(data, factor=2, method="product")

    with pytest.raises(
        ValueError,
        match="'remainder' should be one of 'droptail', 'addtail', 'drophead' or 'addhead'",
    ):
        downsample(data, factor=2, remainder="dropstart")
