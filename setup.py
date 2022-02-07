from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gprof_nn",
    version="0.0.2",
    description="Neural network version of Goddard Profiling Algorithm (GPROF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonpf/gprof_nn",
    author="Simon Pfreundschuh",
    author_email="simon.pfreundschuh@chalmers.se",
    install_requires=[
        "numpy", "scipy", "xarray", "torch", "appdirs", "rich",
        "quantnn>=0.0.4dev"
    ],
    entry_points = {
        'console_scripts': ['gprof_nn=gprof_nn.bin:gprof_nn'],
    },
    packages=find_packages(),
    package_data={
        "gprof_nn": [
            "files/gmi_era5_sensitivities.npy",
            "files/mhs_era5_sensitivities.npy",
            "files/preprocessor_template.pp",
            "files/matplotlib_style.rc",
        ]
    },
    python_requires=">=3.6",
    project_urls={
        "Source": "https://github.com/simonpf/gprof_nn/",
    },
)
