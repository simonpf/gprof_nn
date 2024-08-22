from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version = {}
exec(open("gprof_nn/version.py", "r").read(), version)

setup(
    name="gprof_nn",
    version=version["__version__"],
    description="Neural network version of Goddard Profiling Algorithm (GPROF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonpf/gprof_nn",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=[
        "click", "numpy", "scipy", "xarray", "torch", "appdirs", "rich",
        "quantnn>=0.0.5dev", "h5py", "netCDF4", "h5netcdf", "pandas",
    ],
    extras_require = {
        'development': [
            'pytest', 'pykdtree'
            ],
        'docs': [
            "sphinx", "jupyter-book"
        ]
    },
    entry_points = {
        'console_scripts': ['gprof_nn=gprof_nn.cli:gprof_nn'],
    },
    packages=find_packages(),
    package_data={
        "gprof_nn": [
            "files/gmi_era5_sensitivities.npy",
            "files/mhs_era5_sensitivities.npy",
            "files/preprocessor_template.pp",
            "files/matplotlib_style.rc",
            "files/normalizer_gmi.pckl"
        ]
    },
    python_requires=">=3.6",
    project_urls={
        "Source": "https://github.com/simonpf/gprof_nn/",
    },
)
