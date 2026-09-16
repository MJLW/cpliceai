# docker buildx bake            -> builds both cpliceai_cpu and cpliceai_gpu
# docker buildx bake cpu        -> just the CPU image
# docker buildx bake gpu        -> just the GPU image

variable "TAG" {
  default = "latest"
}

variable "REGISTRY_USER" {
  default = "jhampstead"
}

group "default" {
  targets = ["cpu", "gpu"]
}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
}

target "cpu" {
  inherits = ["_common"]
  args     = { VARIANT = "cpu" }
  tags     = ["${REGISTRY_USER}/cpliceai_cpu", "${REGISTRY_USER}/cpliceai_cpu:${TAG}"]
}

target "gpu" {
  inherits = ["_common"]
  args     = { VARIANT = "gpu" }
  tags     = ["${REGISTRY_USER}/cpliceai_gpu", "${REGISTRY_USER}/cpliceai_gpu:${TAG}"]
}
