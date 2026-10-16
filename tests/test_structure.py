import numpy as np
import pytest
from pytest import approx

from amsterdm.structure import Structure


def test_structure():
    data = np.zeros(10)
    dms = np.arange(3)
    with pytest.raises(ValueError, match="input data is not two-dimensional"):
        Structure(data, dms)

    data = np.zeros((5, 15))
    data[:, 5] = 10
    with pytest.raises(
        ValueError, match="`data` first/outer axis does not match `dms`"
    ):
        Structure(data, dms)

    dms = np.array([2, 1, 0, -1, -2])
    with pytest.raises(ValueError, match="dms should be monotonically increasing"):
        Structure(data, dms)

    dms = dms[::-1]
    structure = Structure(data, dms)

    kc = structure.calc_kc()
    assert kc == approx(5, abs=1e-5)

    with pytest.warns(
        UserWarning,
        match=r"(minimum|maximum) dm estimate is at the (lower|upper) edge of the input dm range",
    ):
        results = structure.calc()

        assert results[0] == approx(-2, abs=1e-5)
