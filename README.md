# Triton Profiling

Triton profiling introduction demo materials and runtime container images.

## Requirements

- Fedora (tested on 42 KDE and Workstation)
- make
- podman

## Images

Container images that provide a runtime environment for either a host with
an NVIDIA GPU or one without.

- cuda                  - Requires an NVIDIA GPU
- nsight                - Does not require an NVIDIA GPU

## Instructions

### Build

#### With an NVIDIA GPU

```bash
make cuda-image
```

#### Without an NVIDIA GPU

```bash
make nsight-image
```

### Runtime

All targets will start a new container and remove it when you exit.

#### Console

Runs the target image and leaves the user inside it at a bash prompt.

```bash
make [cuda | nsight]-console
```

#### Jupyter Notebook

The nsight image can only be used to view the Jupyter notebook,
an NVIDIA GPU and the cuda image are required to run it.

```bash
make [cuda | nsight]-jupyter
```

#### Nsight Systems

Runs the Nsight Systems UI.

```bash
make [cuda | nsight]-systems
```

#### Nsight Compute

Runs the Nsight Compute UI.

```bash
make [cuda | nsight]-compute
```
