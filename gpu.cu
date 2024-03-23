#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
static int bin_per_row; 
int num_bins;

// Initialize long arrays for particle and bins 
int* bin_ids; // store the index of the first particle in this bin 
int* particle_ids;  // stores all the particles in a array 
int* bin_counts; // store the number of particles in each bin

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

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return; 

    particle_t* p = &particles[tid];
    p->ax = p->ay = 0;
    for (int i = 0; i < num_parts; i++) {
        apply_force_gpu(*p, particles[i]);
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

// count the number of particles per bin 
__global__ void count_particles_per_bin(particle_t* particles, int num_parts, int* bin_counts, int bin_per_row, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_x = particles[tid].x / size * bin_per_row;
    int bin_y = particles[tid].y / size * bin_per_row;
    int bin_id = bin_x + bin_y * bin_per_row;
    atomicAdd(&bin_counts[bin_id], 1);
}

// assign particles to bins
__global__ void assign_particles_to_bins(particle_t* particles, int num_parts, int* bin_ids, int* particle_ids, int* bin_counts, int bin_per_row, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_x = particles[tid].x / size * bin_per_row;
    int bin_y = particles[tid].y / size * bin_per_row;
    int bin_id = bin_x + bin_y * bin_per_row;
    int index = atomicAdd(&bin_counts[bin_id], 1);
    bin_ids[bin_id] = index == 0 ? tid : bin_ids[bin_id];
    particle_ids[index] = tid;
}

// initalize the simulation
void init_simulation(particle_t* particles, int num_parts, double size){
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    bin_per_row = ceil(size / cutoff);
    num_bins = bin_per_row * bin_per_row;

    // Allocate memory for bin_ids, particle_ids, and bin_counts
    cudaMalloc((void**)&bin_ids, num_bins * sizeof(int));
    cudaMalloc((void**)&particle_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&bin_counts, num_bins * sizeof(int));
}

// Perform the simulation
void simulate_one_step(particle_t* particles, int num_parts, double size){

    // Reset bin_counts
    cudaMemset(bin_counts, 0, num_bins * sizeof(int));

    // Count the number of particles per bin
    count_particles_per_bin<<<blks, NUM_THREADS>>>(particles, num_parts, bin_counts, bin_per_row, size);

    // Perform exclusive scan on bin_counts
    thrust::device_ptr<int> dev_bin_counts(bin_counts);
    thrust::exclusive_scan(dev_bin_counts, dev_bin_counts + num_bins, dev_bin_counts);

    // Assign particles to bins
    assign_particles_to_bins<<<blks, NUM_THREADS>>>(particles, num_parts, bin_ids, particle_ids, bin_counts, bin_per_row, size);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(particles, num_parts);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(particles, num_parts, size);
}