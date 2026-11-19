import numpy as np
import pytest

from amsterdm.utils import symlog, symlog10, _format_data


def test_symlog():
    array = np.linspace(-np.exp(5), np.exp(5), 20)
    larray = symlog(array)
    # Note how the end points are not exactly 5, sine there is a +1 offset in symlog
    np.testing.assert_allclose(
        [
            -5.00671535,
            -4.8962768,
            -4.77210974,
            -4.63030997,
            -4.46502736,
            -4.26690993,
            -4.01959466,
            -3.69028083,
            -3.19596155,
            -2.17602578,
            2.17602578,
            3.19596155,
            3.69028083,
            4.01959466,
            4.26690993,
            4.46502736,
            4.63030997,
            4.77210974,
            4.8962768,
            5.00671535,
        ],
        larray,
    )

    array = np.logspace(1, 10)
    larray = symlog(array)
    np.testing.assert_allclose(np.log(1 + array), larray)


def test_symlog10():
    array = np.linspace(-1e5, 1e5, 20)
    larray = symlog10(array)
    # Note how the end points are not exactly 5, sine there is a +1 offset in symlog10
    np.testing.assert_allclose(
        [
            -5.36221757,
            -5.31391312,
            -5.25955574,
            -5.1974082,
            -5.12485803,
            -5.03770858,
            -4.92856525,
            -4.78243926,
            -4.56059529,
            -4.08349792,
            4.08349792,
            4.56059529,
            4.78243926,
            4.92856525,
            5.03770858,
            5.12485803,
            5.1974082,
            5.25955574,
            5.31391312,
            5.36221757,
        ],
        larray,
    )


def test_format_data():
    data = {"xx": np.ones((5, 3)), "yy": np.ones((5, 3))}
    with pytest.raises(ValueError, match="missing 'yx' key to match 'yy'"):
        _format_data(data)
    data["xy"] = np.zeros((5, 3))
    data["yx"] = np.zeros((5, 3))
    result = _format_data(data)
    assert result[0].shape == (5, 4, 3)
    assert result[1] == "iquv"

    data = {"i": np.ones((5, 3))}
    result = _format_data(data, convert=False)
    assert isinstance(result[0], dict)
    assert result[1] == "i"
    result = _format_data(data)
    assert result[0].shape == (5, 1, 3)
    assert result[1] == "i"

    data = {
        "i": np.ones((5, 3)),
        "q": np.ones((5, 3)),
        "u": np.ones((5, 3)),
        "v": np.ones((5, 3)),
    }
    result = _format_data(data)
    assert result[0].shape == (5, 4, 3)
    assert result[1] == "iquv"

    data = np.ones((5, 3))
    # Implicit poltype
    result = _format_data(data)
    assert result[0].shape == (5, 1, 3)
    assert result[1] == "i"
    # Explicit poltype
    result = _format_data(data, poltype="i")
    assert result[0].shape == (5, 1, 3)
    assert result[1] == "i"

    data = np.ones((5, 3, 3))
    # Implicit poltype
    result = _format_data(data)
    assert result[0].shape == (5, 3, 3)
    assert result[1] == "iqu"
    # Explicit poltype
    result = _format_data(data, poltype="iqu")
    assert result[0].shape == (5, 3, 3)
    assert result[1] == "iqu"

    data = np.ones((5, 4, 3))
    with pytest.raises(
        ValueError,
        match="polarization type can't be determined; please specify using the 'poltype' keyword",
    ):
        result = _format_data(data)
    # Explicit poltype
    result = _format_data(data, poltype="iquv")
    assert result[0].shape == (5, 4, 3)
    assert result[1] == "iquv"

    data = {"xx": np.ones((5, 3)), "bb": np.ones((5, 3))}
    with pytest.raises(ValueError, match="missing 'yy' key to match 'xx'"):
        _format_data(data)

    data = {"i": np.ones((5, 3)), "q": np.ones((5, 3))}
    with pytest.raises(
        ValueError, match="`poltype` does not match the input dict keys"
    ):
        _format_data(data, poltype="iqu")

    data = {"i": np.ones((5, 3)), "q": np.ones((6, 3))}
    with pytest.raises(ValueError, match="dict values are inconsistent in shape"):
        _format_data(data)

    data = {"i": np.ones((5, 3)), "q": np.ones((5, 1, 3))}
    with pytest.raises(ValueError, match="dict value is not two-dimensional"):
        _format_data(data)

    with pytest.raises(ValueError, match="invalid polarization type"):
        _format_data(data, poltype="xxyy")

    data = {"i": 5}
    with pytest.raises(ValueError, match="dict value is not an array"):
        _format_data(data)

    data = 5
    with pytest.raises(ValueError, match="data is not an array"):
        _format_data(5)

    data = np.ones((5, 2, 1, 3))
    with pytest.raises(ValueError, match="data is not two- or three-dimensional"):
        _format_data(data)

    data = np.ones((5, 5, 3))
    with pytest.raises(ValueError, match="incorrect number of polarization channels"):
        _format_data(data)

    data = np.ones((5, 3))
    result = _format_data(data, poltype="i")
    assert result[0].shape == (5, 1, 3)
    assert result[1] == "i"
    with pytest.raises(ValueError, match="invalid polarization type"):
        result = _format_data(data, poltype="iq")
