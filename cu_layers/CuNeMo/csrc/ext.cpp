#include "gather_features/gather_features.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_features", &GatherFeatures);
  m.def("gather_features_backward", &GatherFeaturesBackward);
}