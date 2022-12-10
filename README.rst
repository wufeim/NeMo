===============
NeMo Extensions
===============

.. image:: https://img.shields.io/pypi/v/nemo.svg
        :target: https://pypi.python.org/pypi/nemo

.. image:: https://img.shields.io/travis/wufeim/nemo.svg
        :target: https://travis-ci.com/wufeim/nemo

.. image:: https://readthedocs.org/projects/nemo/badge/?version=latest
        :target: https://nemo.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Neural mesh models for 3D reasoning.

Features
--------

Models
^^^^^^

- Original NeMo (3D pose estimation)
- 3D aware classification NeMo (3D pose estimation and classification)
- 6D pose estimation NeMo
- (Deformable NeMo ?)

Baselines
^^^^^^^^^

- ResNet50
- Starmap
- Transformer
- Faster RCNN

Datasets
^^^^^^^^

- OOD-CV
- Pascal3D+
- (Synthetic data ?)
- (ObjectNet ?)

Requirements
------------

Python Environment
^^^^^^^^^^^^^^^^^^

1. Create :code:`conda` environment:

.. code::

   conda create -n nemo python=3.9
   conda activate nemo

2. Install :code:`PyTorch` (see `pytorch.org <https://pytorch.org>`_):

.. code::

   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch

3. Install :code:`PyTorch3D` (see `github.com/facebookresearch/pytorch3d <https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md>`_):

.. code::

   conda install -c fvcore -c iopath -c conda-forge fvcore iopath
   conda install -c bottler nvidiacub
   conda install pytorch3d -c pytorch3d

4. Install other dependencies:

.. code::

   conda install numpy matplotlib scipy scikit-image
   conda install pillow
   conda install -c conda-forge timm tqdm pyyaml transformers
   pip install wget BboxTools opencv-python

2. Install NeMo-Extensions:

.. code::

   pip install -e .

Data Preparation
^^^^^^^^^^^^^^^^

See `data/README </data>`_.

TODO
----

- [ ] Add support for VoGe Renderer
- [ ] Add support for multiple GPUs support
- [ ] Add support for some visualizations
