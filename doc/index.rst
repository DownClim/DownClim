.. module:: src.downclim

DownClim |version|
==================
``DownClim`` is an open source project that will make your climate downscaling easily !
It provides a bunch of tools to download and process climate data, and to downscale it.


Description
-----------
The purpose of ``DownClim`` is to offer a tool for regional climate projections including the mechanistic ‘dynamic’ downscaling of the CORDEX initiative. ``DownClim`` is opposed to the direct statistical downscaling of global climate projections found in ``WorldClim`` and ``CHELSA``. The approach is justified by an improvement in regional projections of ``CORDEX`` compared to ``CMIP``, although it can increase uncertainty and sometimes be less reliable. The tool is an automated snakemake workflow easily reproducible and scalable associated to conda environments for enhance reproducibility and portability.

.. _WorldClim: https://www.worldclim.org/
.. _CHELSA: https://chelsa-climate.org/
.. _CORDEX: https://cordex.org/
.. _CMIP: https://www.wcrp-cmip.org/

.. toctree::
    :hidden:

    Home <self>
    Installation <installation>
    Getting Started <getting_started>
    Examples <examples>
    API Reference <reference>

.. container:: button

    :doc: `Installation <installation>` :doc: `Getting Started <getting_started>`
    :doc: `Examples <examples>` :doc: `API Reference <reference>`


| |pypi| |conda-forge| |python| |tests| |coverage| |license| |docs| |black| |mypy| |ruff|

.. |pypi| image:: https://img.shields.io/pypi/v/downclim
    :target: https://pypi.org/project/downclim
    :alt: PyPI version

.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/downclim
    :target: https://anaconda.org/conda-forge/downclim
    :alt: Conda-forge version

.. |tests| image:: https://github.com/DownClim/DownClim/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/DownClim/DownClim/actions/workflows/tests.yml
    :alt: Tests status

.. |coverage| image:: https://codecov.io/gh/DownClim/DownClim/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/DownClim/DownClim
    :alt: Code coverage

.. |license| image:: https://img.shields.io/github/license/DownClim/DownClim
    :target: https://github.com/DownClim/DownClim/blob/main/LICENSE
    :alt: License

.. |docs| image:: https://readthedocs.org/projects/downclim/badge/?version=latest
    :target: https://downclim.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |python| image:: https://img.shields.io/pypi/pyversions/downclim
    :target: https://pypi.org/project/downclim
    :alt: Python versions

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

.. |mypy| image:: https://img.shields.io/badge/types-mypy-blue.svg
    :target: https://github.com/python/mypy
    :alt: Types: mypy

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

Useful links
------------
`Home <https://downclim.github.io/DownClim>`__ |
`Code Repository <https://github.com/DownClim/DownClim>`__ |
`Issues <https://github.com/DownClim/DownClim/issues>`__ |
`Discussions <https://github.com/DownClim/DownClimdiscussions>`__ |
`Releases <https://github.com/DownClim/DownClimreleases>`__ |
