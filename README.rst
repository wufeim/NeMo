====
NeMo
====

.. image:: https://img.shields.io/pypi/v/neural-mesh-model.svg
        :target: https://pypi.python.org/pypi/neural-mesh-model

.. image:: https://readthedocs.org/projects/neural-mesh-model/badge/?version=latest
        :target: https://neural-mesh-model.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Neural mesh models for 3D reasoning.

Features
--------

**Easily train and evaluate neural mesh models for multiple tasks:**

* 3D pose estimation
* 6D poes estimation
* 3D-aware image classification
* Amodal segmenation

**Experiment on various benchmark datasets:**

* PASCAL3D+
* Occluded PASCAL3D+
* ObjectNet3D
* OOD-CV
* SyntheticPASCAL3D+

**Reproduce baseline models for fair comparison:**

* Regression-based models (ResNet50, Faster R-CNN, etc.)
* Transformers
* StarMap

Installation
------------

Environment (manual setup)
^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   pip install git+https://github.com/NVlabs/nvdiffrast/
   pip install wget gdown BboxTools opencv-python xatlas pycocotools seaborn wandb

5. Install NeMo:

.. code::

   pip install -e .

Environment (from `yml`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case the previous method failed, setup the environment from a compiled list of packages:

.. code::

   conda env create -f environment.yml
   pip install git+https://github.com/NVlabs/nvdiffrast/
   pip install -e .

Data Preparation
^^^^^^^^^^^^^^^^

See `data/README </data>`_.

Quick Start
-----------

Train and evaluate a neural mesh model (:code:`NeMo`) on PASCAL3D+ for 3D pose estimation:

.. code::

   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train.py \
       --cate car \
       --config config/pose_estimation_3d_nemo.yaml \
       --save_dir exp/pose_estimation_3d_nemo_car

   CUDA_VISIBLE_DEVICES=0 python3 scripts/inference.py \
       --cate car \
       --config config/pose_estimation_3d_nemo.yaml \
       --save_dir exp/pose_estimation_3d_nemo_car \
       --checkpoint exp/pose_estimation_3d_nemo_car/ckpts/model_800.pth

Train and evaluate a regression-based model (:code:`ResNet50-General`) on PASCAL3D+ for 3D pose estimation:

.. code::

   CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
       --cate all \
       --config config/pose_estimation_3d_resnet50_general.yaml \
       --save_dir exp/pose_estimation_3d_resnet50_general_car

   CUDA_VISIBLE_DEVICES=0 python3 scripts/inference.py \
       --cate car \
       --config config/pose_estimation_3d_resnet50_general.yaml \
       --save_dir exp/pose_estimation_3d_resnet50_general \
       --checkpoint exp/pose_estimation_3d_resnet50_general/ckpts/model_90.pth

Documentation
-------------

See `documentation <https://wufeim.github.io/NeMo/documentation.html>`_.

Roadmap
-------

Models
^^^^^^

- [x] NeMo (Shipped: *Dec 08 2022*)
- [x] NeMo-6D (Shipped: *Dec 09 2022*)
- [x] ResNet50-General (Shipped: *Dec 09 2022*)
- [ ] NeMo-Cls
- [ ] Domain adaptation (from synthetic to real)
- [ ] StarMap
- [ ] PASCAL3D-Specific
- [ ] Faster R-CNN
- [ ] Mask R-CNN
- [ ] Transformers
- [ ] VoGe Renderer

Datasets
^^^^^^^^

- [x] PASCAL3D+ (Shipped: *Dec 06 2022*)
- [x] Occluded PASCAL3D+ (Shipped: *Dec 06 2022*)
- [x] 6D training data (Shipped: *Dec 07 2022*)
- [ ] OOD-CV
- [ ] SyntheticPASCAL3D+
- [ ] ObjectNet3D

Misc
^^^^

- [x] Rewrite training and evaluate entry point (Shipped: *Dec 11 2022*)
- [x] Project page (Shipped: *Dec 11 2022*)
- [ ] Configuration hierarchy
- [ ] Visualization tools
- [ ] Inference demo
- [ ] Save predictions for reuse

Citation
--------

.. code::

   @inproceedings{wang2021nemo,
      title={NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation},
      author={Angtian Wang and Adam Kortylewski and Alan Yuille},
      booktitle={International Conference on Learning Representations},
      year={2021},
      url={https://openreview.net/forum?id=pmj131uIL9H}
   }
   @software{nemo_code_2022,
      title={Neural Mesh Models for 3D Reasoning},
      author={Ma, Wufei and Jesslen, Artur and Wang, Angtian},
      month={12},
      year={2022},
      url={https://github.com/wufeim/NeMo},
      version={1.0.0}
   }

Further Information
-------------------

This repo builds upon several previous works:

* `NeMo: Neural Mesh Models of Contrastive Features for Robust 3D Pose Estimation (ICLR 2021) <https://openreview.net/forum?id=pmj131uIL9H>`_
* `Robust Category-Level 6D Pose Estimation with Coarse-to-Fine Rendering of Neural Features (ECCV 2022) <https://link.springer.com/chapter/10.1007/978-3-031-20077-9_29>`_

Acknowledgements
----------------

In this project, we borrow codes from several other repos:

* :code:`NeMo` by Angtian Wang in `Angtian/NeMo <https://github.com/Angtian/NeMo>`_
* :code:`DMTet` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
* :code:`torch_utils` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
* :code:`uni_rep` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
* :code:`dnnlib` by NVIDIA in `nv-tlabs/GET3D <https://github.com/nv-tlabs/GET3D>`_
