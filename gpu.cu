#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

static int NUM_BLOCKS;
int tot_num_bins;

// Define a struct for linked list nodes
struct ListNode {
    int particle_index;
    ListNode* next;
};

// Initialize arrays for particle ids and bin ids
ListNode** bin_heads;
ListNode** bin_tails;

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

__global__ void compute_forces_gpu(particle_t* parts, ListNode** bin_heads, ListNode** bin_tails, int num_parts, double size, int NUM_BLOCKS) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += steps) {

        // Initialize acceleration to 0
        parts[loc_tid].ax = parts[loc_tid].ay = 0;

        // Get what row and column the particle would be in, with padding
        int dx = (parts[loc_tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[loc_tid].y * NUM_BLOCKS / size) + 1;

        // Iterate through the 3x3 neighboring bins
        for (int m = -1; m <= 1; m++) {
            for (int n = -1; n <= 1; n++) {

                // Get the bin_id of the neighboring bin
                int their_bin_id = dx + m + (NUM_BLOCKS + 2) * (dy + n);

                // Iterate through all the particles in their_bin_id
                ListNode* curr_node = bin_heads[their_bin_id];
                while (curr_node != nullptr) {
                    int particle_j_id = curr_node->particle_index;
                    apply_force_gpu(parts[loc_tid], parts[particle_j_id]);
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

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += steps) {

        particle_t* p = &particles[loc_tid];
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
    for (int loc_tid = tid; loc_tid < num_bins; loc_tid += steps) {
        bin_heads[loc_tid] = nullptr;
        bin_tails[loc_tid] = nullptr;
    }
}

__global__ void insert_into_linked_list(ListNode** bin_heads, ListNode** bin_tails, int* bin_ids, int* particle_indices, int num_parts) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += steps) {
        int bin_id = bin_ids[loc_tid];
        int particle_index = particle_indices[loc_tid];

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

__global__ void compute_bin_ids(particle_t* parts, int* bin_ids, int num_parts, double size, int NUM_BLOCKS) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int steps = blockDim.x * gridDim.x;

    for (int loc_tid = tid; loc_tid < num_parts; loc_tid += steps) {
        // Get what row and column the particle would be in, with padding
        int dx = (parts[loc_tid].x * NUM_BLOCKS / size) + 1;
        int dy = (parts[loc_tid].y * NUM_BLOCKS / size) + 1;
        // Get the bin id of the particle
        int bin_id = dx + (NUM_BLOCKS + 2) * dy;
        bin_ids[loc_tid] = bin_id;
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // num blocks in either x or y direction (+2 in each dimension for padding)
    NUM_BLOCKS = size / cutoff;
    tot_num_bins = (NUM_BLOCKS) * (NUM_BLOCKS);

    // Allocate memory for bin_heads and bin_tails on the GPU
    cudaMalloc((void**)&bin_heads, tot_num_bins * sizeof(ListNode*));
    cudaMalloc((void**)&bin_tails, tot_num_bins * sizeof(ListNode*));

    // Initialize linked lists to nullptr
    initialize_linked_lists<<<blks, NUM_THREADS>>>(bin_heads, bin_tails, tot_num_bins);

    // Allocate memory for particle bin ids and indices on the GPU
    int* d_bin_ids;
    int* d_particle_indices;
     cudaMalloc((void**)&d_bin_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&d_particle_indices, num_parts * sizeof(int));

    // Calculate bin ids for each particle
    compute_bin_ids<<<blks, NUM_THREADS>>>(parts, d_bin_ids, num_parts, size, NUM_BLOCKS);

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

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, bin_heads, bin_tails, num_parts, size, NUM_BLOCKS);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}