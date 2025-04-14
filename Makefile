default: image 

IMAGE_REPO ?= profiling-101
CUDA_NAME ?= cuda
CUDA_RELEASE ?= 12-8
WORKDIR ?= $(PWD)/workdir
CONTAINER_NAME ?= profiling-101

.PHONY: all
all: image run

##@ Container build.
.PHONY: image
image: Containerfile.cuda
	podman build \
	--build-arg "CUDA_RELEASE=$(CUDA_RELEASE)" \
	-t $(IMAGE_REPO)/$(CUDA_NAME):$(CUDA_RELEASE) \
	-f $< .

##@ Container runtime.
.PHONY: nvidia-cdi
nvidia-cdi:
	sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

.PHONY: run
run:
	podman run -it --rm \
	--device nvidia.com/gpu=all \
	--security-opt label=disable \
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
	-v "${WORKDIR}:/workdir:Z" \
	-p 8888:8888 \
	--name $(CONTAINER_NAME) \
	$(IMAGE_REPO)/$(CUDA_NAME):$(CUDA_RELEASE)

.PHONY: console
console:
	podman exec -it $(CONTAINER_NAME) /bin/bash

.PHONY: nsys-ui
nsys-ui:
	podman exec -it $(CONTAINER_NAME) /usr/local/cuda-12.8/bin/nsys-ui

.PHONY: ncu-ui
ncu-ui:
	podman exec -it $(CONTAINER_NAME) /usr/local/cuda-12.8/bin/ncu-ui
