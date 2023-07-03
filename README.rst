====
NeMo
====

.. image:: https://img.shields.io/pypi/v/neural-mesh-model.svg
        :target: https://pypi.python.org/pypi/neural-mesh-model

.. image:: https://readthedocs.org/projects/neural-mesh-model/badge/?version=latest
        :target: https://neural-mesh-model.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

This is the repo for the series works on `Neural Mesh Models <https://arxiv.org/pdf/2101.12378.pdf>`_. In this repo, we implement `3D object pose estimation <https://arxiv.org/pdf/2101.12378.pdf>`_, `6D pose object estimation <https://arxiv.org/pdf/2209.05624.pdf>`_, `object classification <https://arxiv.org/pdf/2305.14668.pdf>`_, and `cross domain training <https://arxiv.org/pdf/2306.00118.pdf>`_. The original implementation of NeMo is `here <https://github.com/Angtian/NeMo>`_.

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
   pip install git+https://github.com/Angtian/VoGE.git
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
       --config config/pose_estimation_3d_runtime.yaml \
       --save_dir exp/pose_estimation_3d_nemo_car

   CUDA_VISIBLE_DEVICES=0 python3 scripts/inference.py \
       --cate car \
       --config config/pose_estimation_3d_runtime.yaml \
       --save_dir exp/pose_estimation_3d_nemo_car \
       --checkpoint exp/pose_estimation_3d_nemo_car/ckpts/model_800.pth

NeMo with VoGE:

.. code::

   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train.py \
       --cate car \
       --config config/pose_estimation_3d_voge.yaml \
       --save_dir exp/pose_estimation_3d_voge_car

   CUDA_VISIBLE_DEVICES=0 python3 scripts/inference.py \
       --cate car \
       --config config/pose_estimation_3d_voge.yaml \
       --save_dir exp/pose_estimation_3d_voge_car \
       --checkpoint exp/pose_estimation_3d_voge_car/ckpts/model_800.pth

NeMo on PASCAL3D+ without scaling during data pre-processing:

.. code::

   CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/train.py \
       --cate car \
       --config config/pose_estimation_3d_runtime_ori.yaml \
       --save_dir exp/pose_estimation_3d_ori_car

   CUDA_VISIBLE_DEVICES=0 python3 scripts/inference.py \
       --cate car \
       --config config/pose_estimation_3d_runtime_ori.yaml \
       --save_dir exp/pose_estimation_3d_ori_car \
       --checkpoint exp/pose_estimation_3d_ori_car/ckpts/model_800.pth

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

Pre-trained Models
-------------

Pre-trained Models for 3D pose estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The pre-trained model for NeMo model:

https://drive.google.com/file/d/14fByOZs_Zzd-97Ulk2BKJhVNFKAnFWvg/view?usp=sharing

+-------+-------+-------+------+--------+------+------+-------+-------+-------+------+-------+-------+-------+
| Cate  | plane | bike  | boat | bottle | bus  | car  | chair | table | mbike | sofa | train | tv    | Mean  |
+=======+=======+=======+======+========+======+======+=======+=======+=======+======+=======+=======+=======+
| Pi/6  | 86.9  | 80.3  | 77.4 | 90.0   | 95.3 | 98.9 | 89.1  | 80.2  | 86.6  | 95.8 | 64.4  | 82.0  | 87.4  |
| Pi/18 | 55.3  | 30.9  | 50.2 | 56.9   | 91.5 | 96.5 | 56.7  | 63.1  | 33.2  | 65.9 | 55.3  | 48.6  | 65.5  |
| Med   | 8.94  | 15.51 | 9.95 | 8.24   | 2.66 | 2.71 | 8.68  | 6.96  | 13.34 | 7.18 | 7.32  | 10.61 | 7.42  |
+-------+-------+-------+------+--------+------+------+-------+-------+-------+------+-------+-------+-------+


The pre-trained model for NeMo-VoGE model:

https://drive.google.com/file/d/1kogFdjVbOIuSlKx1NQ1c1XEjbvJEQWJg/view?usp=sharing

+-------+-------+-------+------+--------+------+------+-------+-------+-------+------+-------+------+-------+
| Cate  | plane | bike  | boat | bottle | bus  | car  | chair | table | mbike | sofa | train | tv   | Mean  |
+=======+=======+=======+======+========+======+======+=======+=======+=======+======+=======+======+=======+
| Pi/6  | 87.8  | 82.9  | 75.4 | 88.2   | 97.4 | 99.0 | 89.5  | 83.6  | 87.4  | 94.4 | 91.3  | 79.5 | 89.4  |
| Pi/18 | 62.3  | 36.7  | 51.0 | 55.2   | 94.5 | 96.4 | 53.4  | 69.7  | 39.1  | 64.1 | 83.3  | 50.2 | 69.1  |
| Med   | 7.57  | 14.02 | 9.7  | 9.1    | 2.38 | 2.89 | 9.27  | 5.7   | 12.3  | 7.47 | 3.84  | 9.9  | 6.88  |
+-------+-------+-------+------+--------+------+------+-------+-------+-------+------+-------+------+-------+


The pre-trained model for NeMo model without scaling:

https://drive.google.com/file/d/1ybVTDx6DvV_H01SUZkKqWQjKu-BfweGJ/view?usp=sharing

+-------+-------+-------+-------+--------+------+------+-------+-------+-------+------+-------+-------+-------+
| Cate  | plane | bike  | boat  | bottle | bus  | car  | chair | table | mbike | sofa | train | tv    | Mean  |
+=======+=======+=======+=======+========+======+======+=======+=======+=======+======+=======+=======+=======+
| Pi/6  | 83.0  | 75.7  | 68.3  | 84.5   | 96.2 | 98.8 | 85.8  | 80.4  | 78.1  | 94.6 | 79.2  | 85.8  | 86.0  |
| Pi/18 | 48.0  | 24.7  | 34.0  | 44.3   | 90.0 | 95.4 | 44.6  | 58.5  | 26.6  | 58.8 | 64.0  | 45.6  | 60.2  |
| Med   | 10.62 | 18.54 | 14.97 | 11.67  | 3.00 | 3.12 | 11.01 | 8.07  | 15.22 | 8.31 | 6.65  | 11.25 | 8.99  |
+-------+-------+-------+-------+--------+------+------+-------+-------+-------+------+-------+-------+-------+




Documentation
-------------

See `documentation <https://wufeim.github.io/NeMo/documentation.html>`_.


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
