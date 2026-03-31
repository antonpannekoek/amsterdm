import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
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


@pytest.fixture(params=[2, 3, 4])
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


def test_downsample_errors():
    data = np.arange(1, 9, dtype=float).reshape(8, 1)
    data = np.repeat(data, 3, axis=1)

    with pytest.raises(ValueError, match="'factor' should be a positive integer"):
        core.downsample(data, factor=0)

    with pytest.raises(ValueError, match="'factor' should be a positive integer"):
        core.downsample(data, factor=1.5)

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

    with pytest.raises(ValueError, match="'factor' should be a positive integer"):
        core.upsample(nddata, 0)
    with pytest.raises(ValueError, match="'factor' should be a positive integer"):
        core.upsample(nddata, 1.5)


def test_correct_bandpass():
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

    corrdata, bkgmean, bkgstd = core.correct_bandpass(data, backgroundrange=[0, 1])
    # Averaged background should match the theoretical value within precision
    np.testing.assert_allclose(bkgmean.mean() / 5, average, rtol=1e-2)
    # After correction, the corrected data has no background and is
    # normally distributed around 0 with sigma = 1
    assert corrdata.mean() == approx(0, abs=1e-3)
    assert corrdata.std() == approx(1, abs=1e-3)

    # Add a single peak value
    data[100, ...] += 10 * bandpass
    corrdata, bkgmean, bkgstd = core.correct_bandpass(data, backgroundrange=[0.1, 1])

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

    sections, bkg = core.findrangelc(data)
    assert sections == [(14, 25)]
    assert bkg[0] == 1  # median background
    assert bkg[1] == 0  # no noise; background identical everywhere

    # Search range shouldn't matter if it completely overlaps
    # the peak range
    sections, _ = core.findrangelc(data, searchrange=[0.0, 0.5])
    assert sections == [(14, 25)]
    # Restricted range
    sections, _ = core.findrangelc(data, searchrange=[0.1, 0.2])
    assert sections == [(14, 20)]
    # Peaks outside of search range
    sections, _ = core.findrangelc(data, searchrange=[0.5, 1])
    assert sections == []

    # Test multiple, non-overlapping ranges
    data[68:73] = 8

    sections, bkg = core.findrangelc(data)
    assert bkg[0] == 1  # median background
    assert bkg[1] == 0  # no noise; background identical everywhere
    assert sections == [(14, 25), (64, 75)]

    # Test different search ranges
    sections, _ = core.findrangelc(data, searchrange=[0.0, 0.5])
    assert sections == [(14, 25)]
    sections, _ = core.findrangelc(data, searchrange=[0.1, 0.2])
    assert sections == [(14, 20)]
    sections, _ = core.findrangelc(data, searchrange=[0.5, 1])
    assert sections == [(64, 75)]
    sections, _ = core.findrangelc(data, searchrange=[0.7, 1])
    assert sections == [(70, 75)]

    # Test multiple overlapping ranges
    # The overlap happens because the actual data
    # is extended slightly
    data[26:30] = 6
    data[77:80] = 4
    sections, bkg = core.findrangelc(data)
    assert bkg[0] == 1  # median background
    assert bkg[1] == 0  # no noise; background identical everywhere
    assert sections == [(14, 32), (64, 82)]


def test_calc_background():
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
        ValueError, match="method should be one of 'mean', 'median', 'mode' or 'none'"
    ):
        core.calc_background(data, method="min")

    backgroundrange = [-1, 1]
    with pytest.raises(ValueError, match="incorrect background range"):
        core.calc_background(data, backgroundrange=backgroundrange)
    backgroundrange = [[0, 0.2], [0.6, 1.1]]
    with pytest.raises(ValueError, match="incorrect background range"):
        core.calc_background(data, backgroundrange=backgroundrange)
    backgroundrange = [0.6, 0.4]
    with pytest.raises(ValueError, match="incorrect background range"):
        core.calc_background(data, backgroundrange=backgroundrange)


def test_dedisperse():
    data = np.ones(80).reshape(10, 8)  # 10 samples by 8 channels
    for i in range(data.shape[0]):
        data[i] = i + 1
    freqs = np.arange(1200, 1000, -25)  # First frequency = top frequency

    # 1 millisecond sampling time

    result = core.dedisperse(data, freqs, tsamp=1, dm=0)
    assert_allclose(result, data)

    result = core.dedisperse(data, freqs, tsamp=1, dm=5)
    # No change at the reference frequency
    assert_allclose(result[:, 0], data[:, 0])
    expected = np.array(
        [
            [1, 2, 2, 3, 4, 5, 5, 6],
            [2, 3, 3, 4, 5, 6, 6, 7],
            [3, 4, 4, 5, 6, 7, 7, 8],
            [4, 5, 5, 6, 7, 8, 8, 9],
            [5, 6, 6, 7, 8, 9, 9, 10],
            [6, 7, 7, 8, 9, 10, 10, 1],
            [7, 8, 8, 9, 10, 1, 1, 2],
            [8, 9, 9, 10, 1, 2, 2, 3],
            [9, 10, 10, 1, 2, 3, 3, 4],
            [10, 1, 1, 2, 3, 4, 4, 5],
        ]
    )
    assert_allclose(result, expected)

    # This wraps around at the lower frequencies
    result = core.dedisperse(data, freqs, tsamp=1, dm=100)
    expected = np.array(
        [
            [1, 3, 7, 1, 6, 2, 9, 8],
            [2, 4, 8, 2, 7, 3, 10, 9],
            [3, 5, 9, 3, 8, 4, 1, 10],
            [4, 6, 10, 4, 9, 5, 2, 1],
            [5, 7, 1, 5, 10, 6, 3, 2],
            [6, 8, 2, 6, 1, 7, 4, 3],
            [7, 9, 3, 7, 2, 8, 5, 4],
            [8, 10, 4, 8, 3, 9, 6, 5],
            [9, 1, 5, 9, 4, 10, 7, 6],
            [10, 2, 6, 10, 5, 1, 8, 7],
        ]
    )
    assert_allclose(result, expected)

    with pytest.raises(
        ValueError, match="`freqs` length does not match the last axis of `data`"
    ):
        freqs = np.arange(1200, 1000, -20)
        core.dedisperse(data, freqs, tsamp=1, dm=100)


def test_flag():
    data = np.ones(80).reshape(10, 8)
    badchannels = [1, 3, 5]
    result = core.flag(data, badchannels)
    expected = np.tile([0, 1, 0, 1, 0, 1, 0, 0], (10, 1)).astype(bool)
    assert_array_equal(result.mask, expected)

    # Add flagged channels to the previous result
    badchannels = {2, 6}
    result = core.flag(result, badchannels)
    expected = np.tile([0, 1, 1, 1, 0, 1, 1, 0], (10, 1)).astype(bool)
    assert_array_equal(result.mask, expected)

    badchannels = [8]
    with pytest.raises(IndexError):
        _ = core.flag(data, badchannels)


def test_calc_intensity():
    # Deprecated function
    pass


def test_create_dynspectrum():
    # four time samples by three channels
    data = {
        "xx": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        "yy": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]) / 2,
    }
    freqs = np.array([1000, 1100, 1200])
    # No background correction
    idata, bkg = core.create_dynspectrum(data, freqs, tsamp=1, dm=0, bkg_method=None)
    expected = np.array(
        [[0.75, 0.75, 0.75], [1.5, 1.5, 1.5], [2.25, 2.25, 2.25], [3, 3, 3]]
    )
    assert_allclose(idata, expected)
    assert_allclose(bkg[0], np.zeros(3), strict=True)
    assert_allclose(bkg[1], np.zeros(3), strict=True)

    # full range for the background calculation
    idata, bkg = core.create_dynspectrum(
        data, freqs, tsamp=1, dm=0, backgroundrange=[0, 1]
    )
    assert_allclose(bkg[0], 1.875 * np.ones(3))
    assert_allclose(bkg[1], 0.883883476 * np.ones(3))
    expected = np.array(
        [
            [-1.341641, -1.341641, -1.341641],
            [-0.447214, -0.447214, -0.447214],
            [0.447214, 0.447214, 0.447214],
            [1.341641, 1.341641, 1.341641],
        ]
    )
    assert_allclose(idata, expected, rtol=1e-6)

    with pytest.raises(
        ValueError, match="'xx' and 'yy' channels do no match in dimensions"
    ):
        data["yy"] = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]) / 2
        core.create_dynspectrum(data, freqs, tsamp=1, dm=0, backgroundrange=None)
    data["yy"] = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]) / 2
    with pytest.raises(
        ValueError, match="`freqs` length does not match the last axis of `data`"
    ):
        freqs = np.array([1000, 1100, 1200, 1300])
        core.create_dynspectrum(data, freqs, tsamp=1, dm=0, backgroundrange=None)
    freqs = np.array([1000, 1100, 1200])

    # Test 3d data
    data = np.array([data["xx"], data["yy"]])
    data = np.rollaxis(data, 1, 0)  # put polarization dimension in the middle
    assert data.shape == (4, 2, 3)

    idata, bkg = core.create_dynspectrum(
        data, freqs, tsamp=1, dm=0, backgroundrange=[0, 1]
    )
    assert_allclose(bkg[0], 1.875 * np.ones(3))
    assert_allclose(bkg[1], 0.883883476 * np.ones(3))
    assert_allclose(idata, expected, rtol=1e-6)

    data = data[:, 0, :]
    idata, bkg = core.create_dynspectrum(
        data, freqs, tsamp=1, dm=0, backgroundrange=[0, 1]
    )
    assert_allclose(bkg[0], [2.5, 2.5, 2.5])
    assert_allclose(bkg[1], np.sqrt([1.25, 1.25, 1.25]))
    expected = np.array(
        [
            [-1.34164079, -1.34164079, -1.34164079],
            [-0.4472136, -0.4472136, -0.4472136],
            [0.4472136, 0.4472136, 0.4472136],
            [1.34164079, 1.34164079, 1.34164079],
        ]
    )
    assert_allclose(idata, expected, rtol=1e-6)

    idata, bkg = core.create_dynspectrum(
        data, freqs, tsamp=1, dm=10, backgroundrange=[0, 1]
    )
    assert_allclose(bkg[0], [2.5, 2.5, 2.5])
    assert_allclose(bkg[1], np.sqrt([1.25, 1.25, 1.25]))
    expected = np.array(
        [
            [-0.4472135954999579, -0.4472135954999579, -1.3416407864998738],
            [0.4472135954999579, 0.4472135954999579, -0.4472135954999579],
            [1.3416407864998738, 1.3416407864998738, 0.4472135954999579],
            [-1.3416407864998738, -1.3416407864998738, 1.3416407864998738],
        ]
    )


def test_calc_lightcurve():
    data = {
        "xx": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),
        "yy": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]) / 2,
    }
    freqs = np.array([1000, 1100, 1200])
    result, bkg = core.calc_lightcurve(
        data, freqs, tsamp=1, dm=10, backgroundrange=[0, 1]
    )
    assert bkg[0] == pytest.approx(1.875)
    assert bkg[1] == pytest.approx(0.883883476483)
    expected = np.array([-2.23606798, 0.4472136, 3.13049517, -1.34164079])
    assert_allclose(result, expected)


def test_calc_lightcurve_from_waterfall():
    # Dedispersed and bandpass corrected data; see test_create_waterfall
    # above.
    data = np.array(
        [
            [-0.4472135954999579, -0.4472135954999579, -1.3416407864998738],
            [0.4472135954999579, 0.4472135954999579, -0.4472135954999579],
            [1.3416407864998738, 1.3416407864998738, 0.4472135954999579],
            [-1.3416407864998738, -1.3416407864998738, 1.3416407864998738],
        ]
    )
    result = core.calc_lightcurve_from_waterfall(data)
    expected = np.array([-2.23606798, 0.4472136, 3.13049517, -1.34164079])
    assert_allclose(result, expected)


def test_bowtie():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    freqs = np.array([1000, 1100, 1200])
    dminterval = [9, 11]
    result = core.bowtie(data, freqs, tsamp=1, dminterval=dminterval, ndm=5)
    expected = np.array(
        [
            [-3.0, 1.66666667, 6.33333333, 11.0],
            [-1.0, 3.66666667, 8.33333333, 5.0],
            [1.0, 5.66666667, 10.33333333, -1.0],
            [3.0, 7.66666667, 4.33333333, 1.0],
            [5.0, 9.66666667, -1.66666667, 3.0],
        ]
    )
    assert_allclose(result, expected)


def test_signal2noise():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    freqs = np.array([1000, 1100, 1200])
    dminterval = [9, 11]
    result = core.signal2noise(data, freqs, 1, dminterval, ndm=9)
    # First value is a list of DMs
    assert_allclose(result[0], np.linspace(9, 11, 9))
    expected = np.array(
        [9.0, 9.0, 9.0, 9.0, 11.0, 8.33333333, 8.33333333, 8.33333333, 7.0]
    )
    # Second value is a list of signal-to-noise ratios
    assert_allclose(result[1], expected)


def test_fit_ratios():
    # Fit a triangle to a Gaussian
    dms = [8, 9, 10, 11, 12]
    ratios = [1, 2, 3, 2, 1]
    ampl, mean, stddev = core.fit_ratios(dms, ratios)

    assert ampl == pytest.approx(2.8402246050285)
    assert mean == pytest.approx(10.0)
    assert stddev == pytest.approx(1.3304348)
