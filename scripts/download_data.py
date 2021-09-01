"""
Script to download files via SFTP.

This is required because the Buffalo disk doesn't support key authentication
and the 'sftp' CLA is essentially useless.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path

from quantnn.files import sftp

LOGGER = logging.getLogger(__file__)

######################################################################
# Arguments
######################################################################

parser = argparse.ArgumentParser(
        description='Download script to download file via SFTP.')
# Input and output data
parser.add_argument('host',
                    metavar='host',
                    type=str,
                    help='The SFTP host from which to download the data.')
parser.add_argument('path',
                    metavar='path',
                    type=str,
                    help='Path of the folder to download.')
parser.add_argument('destination',
                    metavar='destination',
                    type=str,
                    help='The destination where to store the data.')
parser.add_argument('--n_threads',
                    metavar='n',
                    type=int,
                    default=4,
                    help='Number of threads to use for download.')

args = parser.parse_args()
host = args.host
path = args.path
dest = Path(args.destination)
dest.mkdir(exist_ok=True, parents=True)

n_threads = args.n_threads


######################################################################
# Download files
######################################################################


def download_file(host, path, dest):
    """
    Wrapper file to concurrently download files via SFTP.
    """

    with sftp.get_sftp_connection(host) as con:
        destination = Path(dest) / path.name
        LOGGER.info("Downloading file %s to %s.", path, destination)
        try:
            con.get(str(path), destination)
        except Exception:
            os.remove(destination)
    return destination


files = sftp.list_files(host, path)
pool = ThreadPoolExecutor(max_workers=n_threads)
tasks = [pool.submit(download_file, host, f, dest) for f in files]
for t in tasks:
    t.result()
