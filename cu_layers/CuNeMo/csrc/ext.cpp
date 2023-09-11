#include "gather_features/gather_features.h"
#include "mask_weight/mask_weight.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_features", &GatherFeatures);
  m.def("gather_features_backward", &GatherFeaturesBackward);
  m.def("gather_idx", &GatherIdx);
  m.def("mask_weight", &MaskWeight);
}