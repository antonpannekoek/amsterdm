import numpy as np
import pytest
from pytest import approx

from amsterdm.burst import Burst
from amsterdm import core


def test_burst_from_file():
    return
    # path = Path(__file__).parent / "data" / "b_59881.fil"

    # burst = Burst.fromfile(path)


def test_burst():
    return
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
    assert abs(corrdata.mean()) < 1e-3
    assert abs(corrdata.std() - 1) < 1e-3

    # Add a single peak value
    data[100, ...] += 10 * bandpass
    data = rng.normal(loc=5, scale=2, size=(nsamples, nchannels))
    data[100, ...] += 10
    data = data * bandpass[None, :]
    corrdata, bkgmean, bkgstd = core.correct_bandpass(
        data, backgroundrange=[0.1, 1], extra=True
    )
    # Note: normalization by noise (factor 2) results in the signal being 5
    print(corrdata[100, ...].mean())
    assert abs(corrdata[100, ...].mean() - 5) < 0.1
    # assert abs(corrdata[100, ...].mean() - 5) < 1e-3
    # Other values still 0
    # assert abs(corrdata[99, ...].mean()) < 1e-2
    # assert abs(corrdata[101, ...].mean()) < 1e-2
    # Background averaged across first axis = channels leaves 4
    # polarization channels with `average`
    # np.testing.assert_allclose(bkgmean.mean(axis=1), average, rtol=1e-2)
    # print(data.shape, corrdata.shape)
    # print(corrdata[100, ...])
    # Test for the peak without the background
    # np.testing.assert_allclose(corrdata[100, ...], 10)


@pytest.mark.filterwarnings("ignore:'nchans' not found in header")
def test_properties():
    """Test whether properties update on the fly"""

    data = np.ones((50, 20))
    header = {}

    with pytest.raises(
        ValueError, match="'fch1' or related keyword not found in header information"
    ):
        burst = Burst(header, data)

    header["fch1"] = 1400.0  # MHz
    with pytest.raises(
        ValueError, match="'foff' or related keyword not found in header information"
    ):
        burst = Burst(header, data)

    header["foff"] = 4.0  # MHz
    with pytest.raises(
        ValueError, match="'tsamp' or related keyword not found in header information"
    ):
        burst = Burst(header, data)

    header["tsamp"] = 1e-4  # seconds; will be converted into milliseconds
    burst = Burst(header, data)

    assert burst.tsamp == approx(0.1)
    assert burst.nchans == 20  # size of second axis
    assert burst.fch1 == 1400.0
    assert burst.foff == 4.0
    assert burst.fanchor == "mid"  # default is mid-channel

    assert burst.cfreq == approx(1440.0)
    expected = np.arange(1400, 1477, 4)
    np.testing.assert_allclose(burst.freqs, expected)
    # Convert 'times' attribute from days to milliseconds
    expected = np.arange(0, 4.95, 0.1)
    np.testing.assert_allclose(burst.times * 86400 * 1000, expected)

    burst.foff = 3
    expected = np.arange(1400.0, 1458.0, 3.0)
    np.testing.assert_allclose(burst.freqs, expected)

    burst.fch1 = 1200
    expected -= 200.0
    np.testing.assert_allclose(burst.freqs, expected)

    burst.fanchor = "bottom"
    expected -= 1.5
    np.testing.assert_allclose(burst.freqs, expected)
    expected = np.arange(1198.5, 1140, -3)
    burst.foff = -3
    np.testing.assert_allclose(burst.freqs, expected)

    burst.fanchor = "top"
    expected += 3
    np.testing.assert_allclose(burst.freqs, expected)
    burst.foff = 3
    expected = np.arange(1201.5, 1260, 3)
    np.testing.assert_allclose(burst.freqs, expected)

    expected = np.arange(0, 4.95, 0.1)
    np.testing.assert_allclose(burst.reltimes * 1000, expected)

    # Increase sampling time
    burst.tsamp = 0.2
    expected *= 2
    np.testing.assert_allclose(burst.reltimes * 1000, expected)

    # Add 10 days
    burst.tstart = 10
    # Relative times should stay the same
    np.testing.assert_allclose(burst.reltimes * 1000, expected)
    # Absolute times change by 10 days
    expected += 10 * 86400 * 1000
    np.testing.assert_allclose(burst.times * 86400 * 1000, expected)
