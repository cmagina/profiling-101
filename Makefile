IMAGE_REPO ?= triton-profiling
IMAGE_NAME ?= triton-profiling
CUDA_RELEASE ?= 12-8
WORKSPACE ?= $(PWD)/workspace


# Podman Run
define podman_run
	podman run -it --rm \
	--privileged \
	--cap-add=SYS_ADMIN \
	-e DISPLAY=${DISPLAY} \
	-e "WAYLAND_DISPLAY=${WAYLAND_DISPLAY}" \
	-e XDG_RUNTIME_DIR=/tmp \
	-e PULSE_SERVER=${XDG_RUNTIME_DIR}/pulse/native \
	-e QT_QPA_PLATFORM=wayland-egl \
	-e WAYLAND_DISPLAY=wayland-0 \
	-v "${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}:/tmp/${WAYLAND_DISPLAY}:ro" \
	-v "${XDG_RUNTIME_DIR}/pulse:/tmp/pulse:ro" \
	--ipc host \
	-v "${WORKSPACE}:/workspace:Z"
endef


# Podman Runtime CUDA GPU arguments
define cuda_args
	--device nvidia.com/gpu=all \
	--security-opt label=disable
endef


# Podman build
define podman-build
	$(eval $@_IMAGE_NAME_PREFIX = $(1))
	$(eval $@_CONTAINERFILE = $(2))
	podman build \
	--build-arg "CUDA_RELEASE=$(CUDA_RELEASE)" \
	-t $(IMAGE_REPO)/$(IMAGE_NAME)-${$@_IMAGE_NAME_PREFIX}:$(CUDA_RELEASE) \
	-f ${$@_CONTAINERFILE} .
endef


# Container build.
.PHONY: nsight-image
nsight-image: containerfiles/Containerfile.nsight
	@$(call podman-build, "nsight", $<)

.PHONY: cuda-image
cuda-image: containerfiles/Containerfile.cuda
	@$(call podman-build, "cuda", $<)


# Generate the NVIDIA CDI for the container toolkit
.PHONY: nvidia-cdi
nvidia-cdi:
	sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml


# NVIDIA Nsight Tools no-CUDA Support
define nsight-run
	$(podman_run) $(IMAGE_REPO)/$(IMAGE_NAME)-nsight:$(CUDA_RELEASE) $(1)
endef


# Run the NVIDIA Nsight Systems UI (no-CUDA)
.PHONY: nsight-systems
nsight-systems:
	@$(call nsight-run, "nsys-ui")

# Run the NVIDIA Nsight Compute UI (no-CUDA)
.PHONY: nsight-compute
nsight-compute:
	@$(call nsight-run, "ncu-ui")

# Open a shell in the NVIDIA Nsight container (no-CUDA)
.PHONY: nsight-console
nsight-console:
	@$(call nsight-run, "/bin/bash")


# NVIDIA Nsight Tools w/CUDA Support
define cuda-run
	$(podman_run) $(cuda_args) $(IMAGE_REPO)/$(IMAGE_NAME)-cuda:$(CUDA_RELEASE) $(1)
endef


# Run the NVIDIA Nsight Systems UI (w/CUDA)
.PHONY: cuda-systems
cuda-systems:
	@$(call cuda-run, "nsys-ui")

# Run the NVIDIA Nsight Systems UI (w/CUDA)
.PHONY: cuda-compute
cuda-compute:
	@$(call cuda-run, "ncu-ui")


# Run a Jupyter Notebook Server
.PHONY: cuda-jupyter
cuda-jupyter:
	@$(call cuda-run, "start_jupyter")

# Open a shell in the NVIDIA Nsight container (CUDA support)
.PHONY: cuda-console
cuda-console:
	@$(call cuda-run, "/bin/bash")
