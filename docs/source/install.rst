.. _install:

============
Installation
============

Basics
------

First install `Python <https://www.python.org/>`_ >= 3.10, `PyTorch <https://pytorch.org/>`_ >=v.2.0.0, `git <https://git-scm.com/>`_ and `conda <https://github.com/conda/conda>`_. We tested the latest JoeyS2T with

- python 3.11
- torch 2.1.2
- torchaudio 2.1.2
- cuda 12.1

Create and activate a `conda <https://github.com/conda/conda>`_ virtual environment to install the package into:

.. code-block:: bash

   $ conda -n js2t python=3.11
   $ conda activate js2t


Cloning
-------

Then clone JoeyS2T from GitHub and switch to its root directory:

.. code-block:: bash

   (js2t)$ git clone https://github.com/may-/joeys2t.git
   (js2t)$ cd joeys2t

.. note::

    For Windows users, we recommend to doublecheck whether txt files (i.e. ``test/data/toy/*``) have utf-8 encoding.


Installing JoeyNMT
------------------

Install JoeyNMT and its requirements:

.. code-block:: bash

   (js2t)$ python -m pip install -e .

Run the unit tests to make sure your installation is working:

.. code-block:: bash

   (js2t)$ python -m unittest

.. important::

    When running on *GPU* you need to manually install the suitable PyTorch version for your `CUDA <https://developer.nvidia.com/cuda-zone>`_ version. For example, you can install PyTorch 2.1.2 with CUDA v12.1 as follows:

    .. code-block::

        $ python -m pip install --upgrade torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

    This is described in the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

.. note::

    You may need to install extra dependencies (torchaudio backends): `ffmpeg <https://ffmpeg.org/>`_, `sox <https://sox.sourceforge.net/>`_, `soundfile <https://pysoundfile.readthedocs.io/>`_, etc.
    Please refer `torchaudio documentation <https://pytorch.org/audio/stable/installation.html>`_ for details.

You're ready to go!
