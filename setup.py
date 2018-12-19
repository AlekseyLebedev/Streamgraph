import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), 'r', encoding='utf-8').read()


setup(
    name="streamgraph",
    version="0.0.1",
    author="Aleksey Lebedev",
    author_email="leb-al@outlook.com",
    description=("An demostraition of devops"),
    license="MIT",
    keywords="Data visualisation, graphs, stream graph",
    packages=['streamgraph'],
    long_description='Builds the streamgraph as in http://leebyron.com/streamgraph/stackedgraphs_byron_wattenberg.pdf',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Data visualisation",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy'
    ],
)
