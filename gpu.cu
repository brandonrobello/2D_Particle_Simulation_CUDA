#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

static int ROW_BLOCKS;
int binCount;

// Define a struct for linked list nodes
struct ListNode {
    int particle_index;
    ListNode* next;
};

// Initialize arrays for particle ids and bin ids
ListNode** bin_heads;
ListNode** bin_tails;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double del_x = neighbor.x - particle.x;
    double del_y = neighbor.y - particle.y;
    double r2 = del_x * del_x + del_y * del_y;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * del_x;
    particle.ay += coef * del_y;
}

__global__ void compute_forces_gpu(particle_t* particles, ListNode** bin_heads, ListNode** bin_tails, int num_parts, double size, int ROW_BLOCKS) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += steps) {

        // Initialize acceleration to 0
        particles[i].ax = particles[i].ay = 0;

        // Get what row and column the particle would be in
        int del_x = (particles[i].x * ROW_BLOCKS / size) + 1;
        int del_y = (particles[i].y * ROW_BLOCKS / size) + 1;

        // Iterate through the 3x3 neighboring bins
        for (int j = -1; j <= 1; j++) {
            for (int k = -1; k <= 1; k++) {

                // Get the bin_id of the neighboring bin
                int neighbor_id = del_x + j + (ROW_BLOCKS + 2) * (del_y + k);

                // Iterate through all the particles in neighbor_id
                ListNode* curr_node = bin_heads[neighbor_id];
                while (curr_node != nullptr) {
                    int particle_j_id = curr_node->particle_index;
                    apply_force_gpu(particles[i], particles[particle_j_id]);
                    curr_node = curr_node->next;
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += steps) {

        particle_t* p = &particles[i];
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
}

__global__ void initialize_linked_lists(ListNode** bin_heads, ListNode** bin_tails, int num_bins) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    // Initialize bin_heads and bin_tails to nullptr
    for (int i = tid; i < num_bins; i += steps) {
        bin_heads[i] = nullptr;
        bin_tails[i] = nullptr;
    }
}

__global__ void insert_into_linked_list(ListNode** bin_heads, ListNode** bin_tails, int* bin_ids, int* particle_indices, int num_parts) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += steps) {
        int bin_id = bin_ids[i];
        int particle_index = particle_indices[i];

        ListNode* new_node = new ListNode;
        new_node->particle_index = particle_index;
        new_node->next = nullptr;

        if (bin_heads[bin_id] == nullptr) {
            // First node in the bin
            bin_heads[bin_id] = new_node;
            bin_tails[bin_id] = new_node;
        } else {
            bin_tails[bin_id]->next = new_node;
            bin_tails[bin_id] = new_node;
        }
    }
}

__global__ void compute_bin_ids(particle_t* particles, int* bin_ids, int num_parts, double size, int ROW_BLOCKS) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += steps) {
        // Get what row and column the particle would be in
        int del_x = (particles[i].x * ROW_BLOCKS / size) + 1;
        int del_y = (particles[i].y * ROW_BLOCKS / size) + 1;
        // Get the bin id of the particle
        int bin_id = del_x + (ROW_BLOCKS + 2) * del_y;
        bin_ids[i] = bin_id;
    }
}

void init_simulation(particle_t* particles, int num_parts, double size) {
    // This function will be called once before the algorithm begins
    // particles live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // num blocks in either x or y direction
    ROW_BLOCKS = size / cutoff;
    binCount = (ROW_BLOCKS) * (ROW_BLOCKS);

    // Allocate memory for bin_heads and bin_tails on the GPU
    cudaMalloc((void**)&bin_heads, binCount * sizeof(ListNode*));
    cudaMalloc((void**)&bin_tails, binCount * sizeof(ListNode*));

    // Initialize linked lists to nullptr
    initialize_linked_lists<<<blks, NUM_THREADS>>>(bin_heads, bin_tails, binCount);

    // Allocate memory for particle bin ids and indices on the GPU
    int* d_bin_ids;
    int* d_particle_indices;
     cudaMalloc((void**)&d_bin_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&d_particle_indices, num_parts * sizeof(int));

    // Calculate bin ids for each particle
    compute_bin_ids<<<blks, NUM_THREADS>>>(particles, d_bin_ids, num_parts, size, ROW_BLOCKS);

    // Copy particle indices to device
    int* h_particle_indices = new int[num_parts];
    for (int i = 0; i < num_parts; ++i) {
        h_particle_indices[i] = i;
    }
    cudaMemcpy(d_particle_indices, h_particle_indices, num_parts * sizeof(int), cudaMemcpyHostToDevice);

    // Insert particles into linked lists based on bin ids
    insert_into_linked_list<<<blks, NUM_THREADS>>>(bin_heads, bin_tails, d_bin_ids, d_particle_indices, num_parts);

    // Free temporary memory
    cudaFree(d_bin_ids);
    cudaFree(d_particle_indices);
    delete[] h_particle_indices;
}

void simulate_one_step(particle_t* particles, int num_parts, double size) {
    // particles live in GPU memory

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(particles, bin_heads, bin_tails, num_parts, size, ROW_BLOCKS);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(particles, num_parts, size);
}