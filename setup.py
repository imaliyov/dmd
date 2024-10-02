from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dmd',
    version='0.1',
    description="Dynamic mode decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.7.12',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pytest',
        'PyYAML',
        'h5py',
        'pytest-profiling',
        'matplotlib',
        'scipy',
        'tqdm',
    ],
)

