#include "common.h"
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


#define NUM_THREADS 256

// ========================= SAME =============================

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    
    // Use atomic operations to update acceleration
    atomicAdd(&particle.ax, coef * dx);
    atomicAdd(&particle.ay, coef * dy);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

// ========================= SAME =============================

// Put any static global variables here that you will use throughout the simulation.
static int blks, num_bins_per_row, num_bins;
static int *d_bin_ids, *d_particle_ids, *d_bin_offsets, *d_bin_offsets_cp, *d_bin_counts;

// Define bin size
const double bin_size = 1.2 * cutoff;

// Kernel to sort bins
__global__ void assign_bins(int* d_bin_ids, int* d_bin_counts, particle_t* particles, \
                            int num_parts, double size, int num_bins_per_row) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    // Compute bin index based on particle position
    int bx = (int)(particles[tid].x / bin_size);
    int by = (int)(particles[tid].y / bin_size);

    // Ensure bin indices are within range
    bx = min(bx, num_bins_per_row - 1);
    by = min(by, num_bins_per_row - 1);

    int bin_id = bx + by * num_bins_per_row;

    // Atomic increment to track count of particles in each bin
    atomicAdd(&d_bin_counts[bin_id], 1);

    // Store the bin ID along with the original particle index
    d_bin_ids[tid] = bin_id;
}

// Host thrust function to get bin_offsets for scattering particles in bin-order
void compute_prefix_sum(int* d_bin_counts, int* d_bin_offsets, int num_bins) {
    // Perform prefix sum to compute bin start indices
    thrust::exclusive_scan(thrust::device, d_bin_counts, d_bin_counts + num_bins, d_bin_offsets);
    
    // Ensure the last bin offset is correctly set (total number of particles)
    int last_bin_count = 0, last_bin_offset = 0;

    // Synchronize to ensure d_bin_counts is fully updated before reading
    cudaDeviceSynchronize();

    // Copy last bin's count and offset from device to host safely
    cudaMemcpy(&last_bin_count, d_bin_counts + (num_bins - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_bin_offset, d_bin_offsets + (num_bins - 1), sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize(); // Ensure memcpy is completed before using the values

    int total_particles = last_bin_offset + last_bin_count;

    cudaMemcpy(d_bin_offsets + num_bins, &total_particles, sizeof(int), cudaMemcpyHostToDevice);
}

// Kernel to reorder particles in bin-order 
__global__ void scatter_particles(int* d_bin_offsets_cp, int* d_bin_ids, int* d_particle_ids, \
                             int num_parts) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int bin_id = d_bin_ids[tid];

    // Atomic update to determine where to place particle in the new array
    int pos = atomicAdd(&d_bin_offsets_cp[bin_id], 1);

    d_particle_ids[pos] = tid;  // Store particle index in bin-ordered layout
}

// Kernel for bin neighbor lookup in the force computation
__global__ void compute_forces_gpu(particle_t* particles, int* particle_ids, int* bin_offsets,
    int num_parts, int num_bins_per_row) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int p_id = particle_ids[tid];  // Get original particle index in sorted order
    particle_t& particle = particles[p_id];

    int bx = (int)(particle.x / bin_size);
    int by = (int)(particle.y / bin_size);

    // Reset acceleration before applying new forces
    particle.ax = 0.0;
    particle.ay = 0.0;

    // Iterate over this bin and neighboring bins
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
        int nbx = bx + dx, nby = by + dy;
        if (nbx < 0 || nbx >= num_bins_per_row || nby < 0 || nby >= num_bins_per_row) continue;

        int bin_id = nbx + nby * num_bins_per_row;

        // Get start and end indices of the particles in this bin
        int start_idx = bin_offsets[bin_id];
        int end_idx = bin_offsets[bin_id + 1];

        // Loop over all particles in the neighboring bin
        for (int j = start_idx; j < end_idx; j++) {
            int neighbor_id = particle_ids[j];
            if (neighbor_id != p_id) {
                apply_force_gpu(particle, particles[neighbor_id]);
                }
            }
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    num_bins_per_row = ceil(size / bin_size);
    num_bins = num_bins_per_row * num_bins_per_row; 
    
    // Allocate
    cudaMalloc(&d_bin_ids, num_parts * sizeof(int));
    cudaMalloc(&d_particle_ids, num_parts * sizeof(int));
    cudaMalloc(&d_bin_counts, (num_bins * sizeof(int)));
    cudaMalloc(&d_bin_offsets, (num_bins + 1) * sizeof(int));
    cudaMalloc(&d_bin_offsets_cp, (num_bins + 1) * sizeof(int));


    // Initialize
    cudaMemset(d_bin_ids, 0, num_parts * sizeof(int));
    cudaMemset(d_particle_ids, 0, num_parts * sizeof(int));
    cudaMemset(d_bin_counts, 0, (num_bins * sizeof(int)));
    cudaMemset(d_bin_offsets, 0, (num_bins + 1) * sizeof(int));

    cudaDeviceSynchronize();
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Reset bin counts before each step
    cudaMemset(d_bin_counts, 0, num_bins * sizeof(int));

    // 1. Assign particles to bins
    assign_bins<<<blks, NUM_THREADS>>>(d_bin_ids, d_bin_counts, parts, num_parts, size, num_bins_per_row);
    
    cudaDeviceSynchronize();

    // 2. Compute prefix sum of bin_counts to get bin_offsets
    compute_prefix_sum(d_bin_counts, d_bin_offsets, num_bins);

    cudaDeviceSynchronize();  // Ensure prefix sum completes before copying

    // 3. Copy bin_offsets before modifying it in scatter_particles
    cudaMemcpy(d_bin_offsets_cp, d_bin_offsets, (num_bins + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();  // Ensure memory copy completes

    // 4. Scatter particles into the ordered bin layout using copied offsets
    scatter_particles<<<blks, NUM_THREADS>>>(d_bin_offsets_cp, d_bin_ids, d_particle_ids, num_parts);

    cudaDeviceSynchronize();  // Ensure particles are properly scattered

    // 5. Compute forces using bin lookup (using original d_bin_offsets)
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, d_particle_ids, d_bin_offsets, num_parts, num_bins_per_row);

    cudaDeviceSynchronize();  // Ensure forces are computed before moving particles

    // 6. Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}
