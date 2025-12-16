def test_version():
    """Test that the version module has the correct attributes"""

    from amsterdm import version

    assert isinstance(version.version, str)
    assert isinstance(version.__version__, str)
    assert isinstance(version.version_tuple, tuple)
