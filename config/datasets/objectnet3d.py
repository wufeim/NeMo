name: objectnet3d
class_name: nemo.datasets.pascal3d.Pascal3DPlus
root_path: data/ObjectNet3D_NeMo

objectnet3d_raw_path: data/ObjectNet3D
objectnet3d_img_url: ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip
objectnet3d_cad_url: ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip
objectnet3d_anno_url: ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip
objectnet3d_set_url: ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip
objectnet3d_lists_url: https://cs.jhu.edu/~wufeim/ObjectNet3D_lists.zip

dtd_raw_path: data/dtd
mesh_path: data/ObjectNet3D/CAD_single

pad_texture: false
single_mesh: true
training_only: false
evaluation_only: false
augment_by_dist: false
prepare_mode: first
center_and_resize: true
skip_3d_anno: false

image_sizes:
    bed: [640, 800]
    bookshelf: [640, 800]
    calculator: [640, 800]
    cellphone: [640, 800]
    computer: [640, 800]
    cabinet: [640, 800]
    guitar: [640, 800]
    iron: [640, 800]
    knife: [640, 800]
    microwave: [640, 800]
    pen: [640, 800]
    pot: [640, 800]
    rifle: [640, 800]
    slipper: [640, 800]
    stove: [640, 800]
    toilet: [640, 800]
    tub: [640, 800]
    wheelchair: [640, 800]
