"""
Integration tests for gprof_nn package.

This test file runs the retrieval for all released retrieval models.
"""
from gprof_nn.download import download_test_file
from gprof_nn.retrieval import run_retrieval


def test_retrieval_gmi():
    """
    Tests the retrieval for GMI.
    """
    test_file = download_test_file("gmi", "l1c")
    results = run_retrieval(test_file)
    assert len(results) > 0

    test_file = download_test_file("gmi", "preprocessor")
    results = run_retrieval(test_file)
    assert len(results) > 0


def test_retrieval_atms():
    """
    Tests the retrieval for ATMS.
    """
    test_file = download_test_file("atms", "l1c")
    results = run_retrieval(test_file)
    assert len(results) > 0

    test_file = download_test_file("atms", "preprocessor")
    results = run_retrieval(test_file)
    assert len(results) > 0

def test_retrieval_amsr2():
    """
    Tests the retrieval for AMSR2.
    """
    test_file = download_test_file("amsr2", "l1c")
    results = run_retrieval(test_file)
    assert len(results) > 0

    test_file = download_test_file("amsr2", "preprocessor")
    results = run_retrieval(test_file)
    assert len(results) > 0


def test_retrieval_mhs():
    """
    Tests the retrieval for MHS.
    """
    test_file = download_test_file("mhs", "l1c")
    results = run_retrieval(test_file)
    assert len(results) > 0

    test_file = download_test_file("mhs", "preprocessor")
    results = run_retrieval(test_file)
    assert len(results) > 0


def test_retrieval_ssmis():
    """
    Tests the retrieval for SSMIS.
    """
    test_file = download_test_file("ssmis", "l1c")
    results = run_retrieval(test_file)
    assert len(results) > 0

    test_file = download_test_file("ssmis", "preprocessor")
    results = run_retrieval(test_file)
    assert len(results) > 0


def test_retrieval_tmi():
    """
    Tests the retrieval for TMI.
    """
    test_file = download_test_file("tmi", "l1c")
    results = run_retrieval(test_file)
    assert len(results) > 0

    test_file = download_test_file("tmi", "preprocessor")
    results = run_retrieval(test_file)
    assert len(results) > 0
