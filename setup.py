from setuptools import setup, find_packages

setup(
    # Metadata
    name='perf_model',
    version='0.0.1',
    author='Cody Yu, and Xingjian Shi',

    # Package info
    packages=find_packages(where="perf_model"),
    zip_safe=True,
    include_package_data=True,
)
