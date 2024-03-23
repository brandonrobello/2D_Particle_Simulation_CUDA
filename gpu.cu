#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

// static global variables 
int blks; 
int bin_per_row; 
int num_bins;
int bin_size;


int* particle_ids;  // array for sorted particles 
int* bin_ids;       // array containing the index of the first particle that is stored in it 
int* bin_counts;    // array containing the number of particles in each bin


// Function to set an array to zeros 
__global__ void set_to_zero(int* arr, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size)
        return;
    arr[tid] = 0;
}

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
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* particle_ids, int* bin_ids, int* bin_counts, int num_bins, double bin_size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // set the acceleration to zero
    particles[tid].ax = particles[tid].ay = 0;
    // find the bin index for the particles 
    int part_x = particles[tid].x / bin_size;
    int part_y = particles[tid].y / bin_size;

    // loop over the neighboring bins 
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {

            // find the bin index 
            int bin_x = part_x + i;
            int bin_y = part_y + j;

            // check if the bin is within the grid 
            if (bin_x >= 0 && bin_x < num_bins && bin_y >= 0 && bin_y < num_bins) {
                int index = bin_x + bin_y * num_bins;
                int start = bin_ids[index];
                int end = start + bin_counts[index];

                // loop over the particles in the bin 
                for (int k = start; k < end; k++) {
                    apply_force_gpu(particles[tid], particles[particle_ids[k]]);
                }
            }
        }
    }

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

// Function to update the number of particles per bin using atomicAdd 
__global__ void update_bin_counts_gpu(particle_t* parts, int num_parts, int* bin_counts, int num_bins, double bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int part_x = parts[tid].x / bin_size;
    int part_y = parts[tid].y / bin_size;
    int index = part_x + part_y * num_bins;
    atomicAdd(&bin_counts[index], 1);
}

// Function to update particle_ids array 
__global__ void update_particle_ids_gpu(particle_t* parts, int num_parts, int* particle_ids, int* bin_ids, int num_bins, int bin_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int part_x = parts[tid].x / bin_size;
    int part_y = parts[tid].y / bin_size;
    int index = part_x + part_y * num_bins;
    int bin_id = atomicAdd(&bin_ids[index], 1);
    particle_ids[bin_id] = tid;

}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    num_bins = ceil(size / cutoff);
    bin_size = size / num_bins;

    // Allocate memory for particle_ids, bin_ids, and bin_counts
    cudaMalloc(&particle_ids, num_parts * sizeof(int));
    cudaMalloc(&bin_ids, num_bins * num_bins * sizeof(int));
    cudaMalloc(&bin_counts, num_bins * num_bins * sizeof(int));

    // Set particle_ids, bin_ids, and bin_counts to zero
    set_to_zero<<<blks, NUM_THREADS>>>(particle_ids, num_parts);
    set_to_zero<<<blks, NUM_THREADS>>>(bin_ids, num_bins * num_bins);
    set_to_zero<<<blks, NUM_THREADS>>>(bin_counts, num_bins * num_bins);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function
    set_to_zero<<<blks, NUM_THREADS>>>(bin_counts, num_bins * num_bins);
    set_to_zero<<<blks, NUM_THREADS>>>(bin_ids, num_bins * num_bins);

    // Update bin_counts and particle_ids
    update_bin_counts_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, bin_counts, num_bins, bin_size);

    // Perform exclusive scan on bin_counts
    thrust::exclusive_scan(thrust::device, bin_counts, bin_counts + num_bins * num_bins, bin_ids);

    // Copy the result back to bin_counts
    cudaMemcpy(bin_counts, bin_ids, num_bins * num_bins * sizeof(int), cudaMemcpyDeviceToDevice);

    // Update particle_ids
    update_particle_ids_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids, bin_ids, num_bins, bin_size);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particle_ids, bin_ids, bin_counts, num_bins, bin_size);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}