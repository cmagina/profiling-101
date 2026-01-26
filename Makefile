IMAGE_REPO ?= triton-profiling
WORKSPACE ?= $(PWD)/workspace
CUDA_EPEL_VERSION ?= 9
CUDA_IMAGE_NAME ?= cuda-profiling
CUDA_VERSION ?= 12-8
CUDA_PYTHON_VERSION ?= 3.12
CUDA_VISIBLE_DEVICES ?= 0
CUDA_WORKSPACE ?= ${WORKSPACE}/CUDA
ROCM_EPEL_VERSION ?= 10
ROCM_IMAGE_NAME ?= rocm-profiling
ROCM_VERSION ?= 7.0.3
ROCM_PYTHON_VERSION ?= 3.12
ROCM_WORKSPACE ?= ${WORKSPACE}/ROCm
ROCR_VISIBLE_DEVICES ?= 0
CTR_CMD := $(or $(shell command -v podman), $(shell command -v docker))

# Container Run
define ctr_run
	$(CTR_CMD) run -it --rm \
	$(1) \
	$(IMAGE_REPO)/$(2) \
	$(3)
endef

# CUDA runtime arguments
ifeq ($(CTR_CMD),/usr/bin/podman)
	cuda_dev_args=--device nvidia.com/gpu=all
else ifeq ($(CTR_CMD),/usr/bin/docker)
	cuda_dev_args=--runtime=nvidia --gpus all
endif

define cuda_args
	--security-opt label=disable \
	--privileged \
	--cap-add=SYS_ADMIN \
	-e CUDA_VERSION=$(CUDA_VERSION) \
	-e CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)
endef

define cuda_vol
	-v "${CUDA_WORKSPACE}:/workspace/user:Z"
endef

# Runtime Nsight arguments
define nsight_args
	-e DISPLAY=${DISPLAY} \
	-e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
	-e XDG_RUNTIME_DIR=/tmp \
	-e PULSE_SERVER=${XDG_RUNTIME_DIR}/pulse/native \
	-e QT_QPA_PLATFORM=wayland-egl \
	-e WAYLAND_DISPLAY=wayland-0 \
	-v "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}:/tmp/${WAYLAND_DISPLAY}:ro" \
	-v "${XDG_RUNTIME_DIR}/pulse:/tmp/pulse:ro"
endef

# ROCm runtime arguments
define rocm_args
	--device=/dev/kfd \
	--device=/dev/dri \
	--cap-add=SYS_PTRACE \
	--group-add keep-groups \
	--ipc=host \
	--security-opt seccomp=unconfined \
	-e ROCM_VERSION=$(ROCM_VERSION) \
	-e ROCR_VISIBLE_DEVICES=$(ROCR_VISIBLE_DEVICES)
endef

define rocm_vol
	-v "${ROCM_WORKSPACE}:/workspace/user:Z"
endef

# Runtime Jupyter notebook arguments
define notebook_args
	-e NOTEBOOK_PORT=$(NOTEBOOK_PORT) \
	-e NOTEBOOK_DIR=/workspace/user \
	-p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT)
endef


# Build NVIDIA Nsight profiling image
.PHONY: cuda-image
cuda-image: NOTEBOOK_PORT ?= 8888
cuda-image: containerfiles/Dockerfile.cuda
	$(CTR_CMD) build \
	--build-arg "CUDA_VERSION=$(CUDA_VERSION)" \
	--build-arg "EPEL_VERSION=$(CUDA_EPEL_VERSION)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	--build-arg "PYTHON_VERSION=$(CUDA_PYTHON_VERSION)" \
	-t $(IMAGE_REPO)/$(CUDA_IMAGE_NAME):$(CUDA_VERSION) \
	-f $< .


# Run the NVIDIA Nsight Systems UI (with no-GPU support)
.PHONY: nsight-systems
nsight-systems: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),nsys-ui)

# Run the NVIDIA Nsight Compute UI (with no-GPU support)
.PHONY: nsight-compute
nsight-compute: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),ncu-ui)

# Run a Jupyter Notebook Server (with no-GPU support)
.PHONY: nsight-jupyter
nsight-jupyter: NOTEBOOK_PORT ?= 8888
nsight-jupyter: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),start_jupyter)

# Open a shell in the NVIDIA Nsight container (with no-GPU support)
.PHONY: nsight-console
nsight-console: NOTEBOOK_PORT ?= 8889
nsight-console: cuda-image
	$(call ctr_run,$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),)


# Run the NVIDIA Nsight Systems UI (with GPU support)
.PHONY: cuda-systems
cuda-systems: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),nsys-ui)

# Run the NVIDIA Nsight Systems UI (with GPU support)
.PHONY: cuda-compute
cuda-compute: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),ncu-ui)

# Run a Jupyter Notebook Server (with GPU support)
.PHONY: cuda-jupyter
cuda-jupyter: NOTEBOOK_PORT ?= 8888
cuda-jupyter: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),start_jupyter)

# Open a shell in the NVIDIA Nsight container (with GPU support)
.PHONY: cuda-console
cuda-console: NOTEBOOK_PORT ?= 8889
cuda-console: cuda-image
	$(call ctr_run,$(cuda_dev_args)$(cuda_args)$(cuda_vol)$(nsight_args)$(notebook_args),$(CUDA_IMAGE_NAME):$(CUDA_VERSION),)


.PHONY: rocm-image
rocm-image: NOTEBOOK_PORT ?= 8888
rocm-image: containerfiles/Dockerfile.rocm
	$(CTR_CMD) build \
	--build-arg "EPEL_VERSION=$(ROCM_EPEL_VERSION)" \
	--build-arg "ROCM_VERSION=$(ROCM_VERSION)" \
	--build-arg "NOTEBOOK_PORT=$(NOTEBOOK_PORT)" \
	--build-arg "PYTHON_VERSION=$(ROCM_PYTHON_VERSION)" \
	-t $(IMAGE_REPO)/$(ROCM_IMAGE_NAME):$(ROCM_VERSION) \
	-f $< .


# Run a Jupyter Notebook Server (with GPU support)
.PHONY: rocm-jupyter
rocm-jupyter: NOTEBOOK_PORT ?= 8888
rocm-jupyter: rocm-image
	$(call ctr_run,$(rocm_args)$(rocm_vol)$(notebook_args),$(ROCM_IMAGE_NAME):$(ROCM_VERSION),start_jupyter)

# Open a shell in the AMD ROCm container (with GPU support)
.PHONY: rocm-console
rocm-console: NOTEBOOK_PORT ?= 8889
rocm-console: rocm-image
	$(call ctr_run,$(rocm_args)$(rocm_vol)$(notebook_args),$(ROCM_IMAGE_NAME):$(ROCM_VERSION),)

.PHONY: clean
clean:
	rm -rf workspace/.ipynb_checkpoints
