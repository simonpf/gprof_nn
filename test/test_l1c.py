"""
Test reading and manipulation of L1C files.
"""
from pathlib import Path

from gprof_nn.data.l1c import L1CFile

def test_find_granule():
    """
    Test finding of specific L1C file and reading data into
    xarray.Dataset.
    """
    l1c_path = Path(__file__).parent / "data"

    l1c_file = L1CFile.open_granule(27510, l1c_path)
    l1c_data = l1c_file.to_xarray_dataset()

    assert l1c_data.pixels.size == 221
    assert l1c_data.scans.size == 2962
