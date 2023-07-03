Data Preparation
================

PASCAL3D+ and Occluded PASCAL3D+
--------------------------------

.. code::

   python3 prepare_pascal3d.py \
       --config config/datasets/pascal3d_runtime.yaml

Prepare data without centering and resize:

.. code::

   python3 prepare_pascal3d.py \
       --config config/datasets/pascal3d_ori.yaml


**Parameters.** The parameters are loaded from the :code:`.yaml` files.

* :code:`pad_texture`: If :code:`True`, use describable textures when padding.
* :code:`occ_levels`: The occlusion levels we prepare for training and validation data.
* :code:`single_mesh`: Type of mesh we generate.
* :code:`root_path`: Path to the generated data.
* :code:`training_only`: If :code:`True`, skip validation data.
* :code:`image_sizes`: Image sizes of the output images.
* :code:`mesh_path`: Path to the meshes used for generating 3D keypoint annotations.
* :code:`prepare_mode`: Preparation mode, :code:`first` or :code:`all`.
* :code:`augment_by_dist`: If :code:`True`, augment samples by object distances (scales); commonly used for 6D pose estimation training.

OOD-CV
------

.. code::

   python3 prepare_ood_cv.py \
       --config config/datasets/ood_cv.yaml

**Parameters.** The parameters are loaded from the :code:`.yaml` files.

* :code:`nuisances`: Types of nuisances we consider.
* :code:`pad_texture`: If :code:`True`, use describable textures when padding.
* :code:`occ_levels`: The occlusion levels we prepare for training and validation data.
* :code:`single_mesh`: Type of mesh we generate.
* :code:`root_path`: Path to the generated data.
* :code:`training_only`: If :code:`True`, skip validation data.
* :code:`image_sizes`: Image sizes of the output images.
* :code:`mesh_path`: Path to the meshes used for generating 3D keypoint annotations.
* :code:`prepare_mode`: Preparation mode, :code:`first` or :code:`all`.
* :code:`augment_by_dist`: If :code:`True`, augment samples by object distances (scales); commonly used for 6D pose estimation training.

Synthetic ShapeNet
------------------

First download ShapeNet v1 from `shapenet.org <https://shapenet.org>`_. Then install `Blender 2.90 <https://download.blender.org/release/Blender2.90/>`_:

.. code::

   apt-get install -y libxi6 libgconf-2-4 libfontconfig1 libxrender1
   wget https://download.blender.org/release/Blender2.90/blender-2.90.0-linux64.tar.xz
   tar -xf blender-2.90.0-linux64.tar.xz
   cd blender-2.90.0-linux64/2.90/python/bin
   ./python3.7m -m ensurepip
   ./python3.7m -m pip install numpy

Then run the following script.

.. code::

   python3 create_synthetic_shapenet.py \
       --config config/datasets/synthetic_shapenet.yaml

Misc
----

**Tetrahedra grids.**

.. code::

   wget https://www.cs.jhu.edu/~wufeim/NeMo/tets.zip
   unzip tets.zip
   rm tets.zip
