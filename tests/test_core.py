import numpy as np
from numpy.testing import assert_allclose
import pytest
from pytest import approx

from amsterdm import core


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
    def test_downsample(self, nddata, remainder, factor, method, means):
        result = core.downsample(
            nddata, factor=factor, remainder=remainder, method=method
        )

        # Build expected array from the list of means
        expected = np.asarray(
            [m * np.ones(nddata.shape[1:], dtype=float) for m in means]
        )
        assert_allclose(result, expected, strict=True)


#    def test_upsample(nddata, remainder, factor, method, means):
#        result = core.upsample(nddata, factor=factor)


def test_downsample_errors():
    data = np.arange(1, 9, dtype=float)

    with pytest.raises(ValueError, match="'data' should at least be two-dimensional"):
        core.downsample(data, factor=2)

    data = data.reshape(8, 1)
    data = np.repeat(data, 3, axis=1)

    with pytest.raises(ValueError, match="'factor' should be a positive integer"):
        core.downsample(data, factor=0)

    with pytest.raises(ValueError, match="'method' should be one of 'mean' or 'sum'"):
        core.downsample(data, factor=2, method="product")

    with pytest.raises(
        ValueError,
        match="'remainder' should be one of 'droptail', 'addtail', 'drophead' or 'addhead'",
    ):
        core.downsample(data, factor=2, remainder="dropstart")


@pytest.mark.parametrize("factor", [2, 3, 4])
def test_upsample(nddata, factor):
    result = core.upsample(nddata, factor=factor)
    expected = np.repeat(nddata, factor, axis=0)
    assert_allclose(result, expected, strict=True)


def test_bandpass():
    """Test basic bandpass correction"""

    # Note that the tolerances are relatively loose, and
    # are specific to the random seed

    nsamples, nchannels = 8192, 256
    bandpass = 1 + np.sin(np.linspace(0, np.pi, nchannels))
    # theoretical value; actual value limited by number of channels
    average = 1 - 2 * np.cos(np.pi) / np.pi
    # Set a background with a value of 5 and noise of 2
    rng = np.random.default_rng(seed=0)
    data = rng.normal(loc=5, scale=2, size=(nsamples, nchannels))
    data = data * bandpass[None, :]
    # Test with a low tolerance
    np.testing.assert_allclose(data.mean() / 5, average, rtol=1e-2)

    corrdata, bkgmean, bkgstd = core.correct_bandpass(
        data, backgroundrange=[0, 1], extra=True
    )
    # Averaged background should match the theoretical value within precision
    np.testing.assert_allclose(bkgmean.mean() / 5, average, rtol=1e-2)
    # After correction, the corrected data has no background and is
    # normally distributed around 0 with sigma = 1
    assert corrdata.mean() == approx(0, abs=1e-3)
    assert corrdata.std() == approx(1, abs=1e-3)

    # Add a single peak value
    data[100, ...] += 10 * bandpass
    corrdata, bkgmean, bkgstd = core.correct_bandpass(
        data, backgroundrange=[0.1, 1], extra=True
    )

    # Note: normalization by noise (factor 2) results in the signal being 5
    assert corrdata[100, ...].mean() == approx(5, abs=0.2)
    # The median is often better, though not much here
    assert corrdata[100, ...].mean() == approx(5, abs=0.1)
    # Verify that surrounding channels are still near 0
    assert corrdata[99, ...].mean() == approx(0, abs=0.1)
    assert corrdata[101, ...].mean() == approx(0, abs=0.1)


def test_findpeaklc():
    """Test the peak finding for light curve (one-dimensional) data

    This is simply `np.argmax` behind the scenes

    """

    data = np.ones(100, dtype=float)
    # Test peak at the edges
    data[0] = 5
    index = core.findpeaklc(data)
    assert index == 0
    data[0] = 1
    data[-1] = 5
    index = core.findpeaklc(data)
    assert index == 99
    # With two identical peaks (very unlikely in practice)
    # only return the first peak index
    data[20] = 5
    index = core.findpeaklc(data)
    assert index == 20
    data[0] = data[-1] = 1
    data[20] = 5
    data[90] = 3
    index = core.findpeaklc(data)
    assert index == 20
    index = core.findpeaklc(data, searchrange=(0, 0.5))
    assert index == 20
    # Select only the last 20%; finds the second peak
    index = core.findpeaklc(data, searchrange=[0.8, 1])
    assert index == 90
    # Finds no peak; returns first index in the search range
    # since the background is the same everywhere
    index = core.findpeaklc(data, searchrange=[0.5, 0.8])
    assert index == 50


def test_findrangelc():
    """ """

    data = np.ones(100, dtype=float)

    data[18:23] = 5

    sections = core.findrangelc(data)
    assert sections == [(14, 25)]
    # Search range shouldn't matter if it completely overlaps
    sections = core.findrangelc(data, searchrange=[0.0, 0.5])
    assert sections == [(14, 25)]
    sections = core.findrangelc(data, searchrange=[0.1, 0.2])
    assert sections == [(14, 20)]
    sections = core.findrangelc(data, searchrange=[0.5, 1])
    assert sections == []

    # Test multiple, non-overlapping ranges
    data[68:73] = 8

    sections = core.findrangelc(data)
    assert sections == [(14, 25), (64, 75)]
    # Search range shouldn't matter if it completely overlaps
    sections = core.findrangelc(data, searchrange=[0.0, 0.5])
    assert sections == [(14, 25)]
    sections = core.findrangelc(data, searchrange=[0.1, 0.2])
    assert sections == [(14, 20)]
    sections = core.findrangelc(data, searchrange=[0.5, 1])
    assert sections == [(64, 75)]
    sections = core.findrangelc(data, searchrange=[0.7, 1])
    assert sections == [(70, 75)]

    # Test multiple overlapping ranges
    # The overlap happens because the actual data
    # is extended slightly
    data[26:30] = 6
    data[77:80] = 4
    sections = core.findrangelc(data)
    assert sections == [(14, 32), (64, 82)]


def test_background_calculation():
    nsamples, nchannels = 1024, 128
    rng = np.random.default_rng(seed=0)
    data = rng.normal(loc=5, scale=2, size=(nsamples, nchannels))

    mean, std = core.calc_background(data)
    assert mean.mean() == approx(5, abs=1e-2)
    assert std.mean() == approx(2, abs=1e-2)

    # Note: the default background range is [[0, 0.333], [0.667, 1]]
    backgroundrange = [[0, 0.2], [0.8, 1]]
    mean, std = core.calc_background(data, backgroundrange=backgroundrange)
    assert mean.mean() == approx(5, abs=1e-2)
    assert std.mean() == approx(2, abs=1e-2)

    # Ignore data in the non-background interval
    data[400:600, :] = 10
    mean, std = core.calc_background(data, backgroundrange=backgroundrange)

    # Contrast with background calculated over the full range
    mean, std = core.calc_background(data, backgroundrange=[0, 1])
    assert mean.mean() == approx(5.6275, abs=1e-5)
    assert std.mean() == approx(2.67565, abs=1e-5)
    mean, std = core.calc_background(data, backgroundrange=[0, 1], method="mean")
    assert mean.mean() == approx(5.976, abs=1e-5)
    # Note how the standard deviation uses the mean, also for the
    # default "median" method
    assert std.mean() == approx(2.67565, abs=1e-5)

    with pytest.raises(
        ValueError, match="method should be one of 'mean', 'median' or 'mode'"
    ):
        core.calc_background(data, method="min")
