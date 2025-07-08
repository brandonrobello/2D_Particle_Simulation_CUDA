# 2D Particle Simulation Using CUDA

## Contributions

* **Write-Up**: All
* **Brandon**: CUDA design and implementation; timing bottleneck breakdown
* **Kevin**: Benchmarking performance

---

## Introduction

In Homework 2.3, we optimized a 2D repulsive particle simulation using **CUDA** on an NVIDIA GPU.

### Key Goals:

* Replace naive `O(n²)` pairwise interactions with `O(n)` binning
* Leverage massive parallelism from GPU threads for scalable simulation
* Design memory-efficient data structures and maintain synchronization

---

## GPU Optimization Strategies

### Spatial Binning on the GPU

To reduce redundant force calculations, we use spatial bins where each particle only interacts with its bin and neighboring bins.

### Core Data Structures

1. `d_bin_ids`: Bin ID for each particle
2. `d_bin_counts`: Count of particles per bin
3. `d_particle_ids`: Particle indices arranged by bin
4. `d_bin_offsets`: Prefix-sum array for bin start indices

### CUDA Kernels and Execution Flow

* `assign_bins`: Each thread computes a particle’s bin and updates count using `atomicAdd`
* `compute_prefix_sum`: Thrust's `exclusive_scan` to generate bin offsets
* `scatter_particles`: Sort particles into contiguous bin order
* `compute_forces_gpu`: Calculate forces only for nearby particles
* `move_gpu`: Update particle positions

### Memory Management

* `cudaDeviceSynchronize()` ensures kernel order and correctness
* `cudaMemcpy` and `cudaMemset` are used between steps for accurate data movement

---

## Runtime Performance

### Scaling

* **Naive serial** approach: `O(n²)`
* **Optimized CUDA** implementation: \~`O(n)` due to binning and parallelism

### Observations

* **For large particle counts (>100K)**, CUDA **significantly outperforms** OpenMP (64 threads) and MPI (2 nodes × 64 ranks)
* **MPI outperforms CUDA at small particle counts** due to underutilization of GPU resources

---

## Synchronization & Timing Breakdown

We categorize execution time into three major components:

### 1. Computation

* `compute_forces_gpu()`
* `assign_bins()`
* `move_gpu()`

### 2. Communication

* Memory copy: `cudaMemcpy`, `cudaMemset`
* `compute_prefix_sum()` via Thrust
* `scatter_particles()` to reorder particle indices

### 3. Synchronization

* `cudaDeviceSynchronize()` to enforce execution ordering

### Runtime Breakdown Summary

* **At small scales**: Communication and synchronization dominate
* **At large scales**: Computation (esp. force calculation) dominates
* **Scatter time increases** with particles due to sorting of indices

---

## Comparison with HW2-1 and HW2-2

| Method     | Best At Small N? | Best At Large N? | Notes                                  |
| ---------- | ---------------- | ---------------- | -------------------------------------- |
| **OpenMP** | ❌                | ❌                | 64 threads — lowest performance        |
| **MPI**    | ✅                | ⚠️ (plateaus)    | Good initially, bottlenecks after 100K |
| **CUDA**   | ⚠️               | ✅                | Becomes best for large N               |

---

## Conclusion

We successfully implemented a GPU-accelerated particle simulator using CUDA that:

* Scales efficiently for large problem sizes
* Achieves near-linear performance with binning and coalesced memory
* Highlights changing bottlenecks: communication → computation as `N` increases

### Future Work:

* Use shared memory to further reduce global memory accesses
* Explore alternative sorting strategies for faster `scatter_particles()`
* Investigate persistent kernel launches for further speedup
