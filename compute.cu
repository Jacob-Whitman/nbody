/*
 * CUDA implementation of n-body gravitational force computation
 * 
 * Memory flow:
 * - Host arrays hPos, hVel, mass are the source of truth
 * - Device arrays d_hPos, d_hVel, d_mass are allocated in nbody.c
 * - Before the time loop, data is copied from host to device
 * - Each compute() call operates entirely on device arrays
 * - After the time loop, final state is copied back to host
 * 
 * Kernel structure:
 * 1. computePairwiseAccelerations: Each thread computes acceleration on body i due to body j
 *    - Uses 2D grid: blockIdx.y*blockDim.y + threadIdx.y = i, blockIdx.x*blockDim.x + threadIdx.x = j
 *    - Stores result in d_accels[i*NUMENTITIES + j]
 * 2. sumAccelerationsAndUpdate: Each thread handles one body i
 *    - Sums all accelerations for body i from d_accels
 *    - Updates velocity: v_new = v_old + a * INTERVAL
 *    - Updates position: p_new = p_old + v_new * INTERVAL
 * 
 * Shared memory optimization:
 * - Uses tiled approach with TILE_SIZE bodies per tile (default 16)
 * - Each block loads a tile of positions and masses into shared memory
 * - Reduces global memory traffic by reusing loaded data across threads
 * - Accumulates partial acceleration contributions in shared memory, then writes to global
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

// Shared memory tile size for optimization
#define TILE_SIZE 16

/*
 * Kernel 1: Compute pairwise accelerations using global memory
 * Each thread computes the acceleration on body i due to body j
 * Grid: 2D grid where thread (i, j) computes accels[i][j]
 */
__global__ void computePairwiseAccelerations(vector3 *d_accels, vector3 *d_hPos, double *d_mass, int numEntities) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numEntities || j >= numEntities) return;
    
    if (i == j) {
        // Self-interaction: zero acceleration
        d_accels[i * numEntities + j][0] = 0.0;
        d_accels[i * numEntities + j][1] = 0.0;
        d_accels[i * numEntities + j][2] = 0.0;
    } else {
        // Compute distance vector: r_i - r_j
        vector3 distance;
        distance[0] = d_hPos[i][0] - d_hPos[j][0];
        distance[1] = d_hPos[i][1] - d_hPos[j][1];
        distance[2] = d_hPos[i][2] - d_hPos[j][2];
        
        // Compute magnitude squared and magnitude
        double magnitude_sq = distance[0] * distance[0] + 
                              distance[1] * distance[1] + 
                              distance[2] * distance[2];
        double magnitude = sqrt(magnitude_sq);
        
        // Acceleration magnitude: -G * m_j / r^2
        double accelmag = -1.0 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
        
        // Acceleration vector: (accel_mag / r) * distance
        d_accels[i * numEntities + j][0] = accelmag * distance[0] / magnitude;
        d_accels[i * numEntities + j][1] = accelmag * distance[1] / magnitude;
        d_accels[i * numEntities + j][2] = accelmag * distance[2] / magnitude;
    }
}

/*
 * Kernel 2: Sum accelerations and update velocities/positions
 * Each thread handles one body i
 */
__global__ void sumAccelerationsAndUpdate(vector3 *d_accels, vector3 *d_hVel, vector3 *d_hPos, int numEntities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= numEntities) return;
    
    // Sum all accelerations for body i
    vector3 accel_sum = {0.0, 0.0, 0.0};
    for (int j = 0; j < numEntities; j++) {
        accel_sum[0] += d_accels[i * numEntities + j][0];
        accel_sum[1] += d_accels[i * numEntities + j][1];
        accel_sum[2] += d_accels[i * numEntities + j][2];
    }
    
    // Update velocity: v_new = v_old + a * INTERVAL
    d_hVel[i][0] += accel_sum[0] * INTERVAL;
    d_hVel[i][1] += accel_sum[1] * INTERVAL;
    d_hVel[i][2] += accel_sum[2] * INTERVAL;
    
    // Update position: p_new = p_old + v_new * INTERVAL
    d_hPos[i][0] += d_hVel[i][0] * INTERVAL;
    d_hPos[i][1] += d_hVel[i][1] * INTERVAL;
    d_hPos[i][2] += d_hVel[i][2] * INTERVAL;
}

/*
 * Shared memory optimized kernel: Compute pairwise accelerations using tiled approach
 * Each block processes a TILE_SIZE x TILE_SIZE tile of the acceleration matrix.
 * Positions and masses are loaded into shared memory and reused across threads in the block,
 * reducing global memory reads from O(n^2) to O(n^2 / TILE_SIZE) per tile.
 */
__global__ void computePairwiseAccelerationsShared(vector3 *d_accels, vector3 *d_hPos, double *d_mass, int numEntities) {
    // Shared memory for tiles of positions and masses
    __shared__ vector3 shared_pos_i[TILE_SIZE];
    __shared__ vector3 shared_pos_j[TILE_SIZE];
    __shared__ double shared_mass_j[TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Compute which tile this block is processing
    int tile_i = blockIdx.y;
    int tile_j = blockIdx.x;
    
    // Global indices for this thread
    int i = tile_i * TILE_SIZE + ty;
    int j = tile_j * TILE_SIZE + tx;
    
    // Load tile of i-bodies into shared memory
    if (ty < TILE_SIZE) {
        int load_i = tile_i * TILE_SIZE + ty;
        if (load_i < numEntities) {
            shared_pos_i[ty][0] = d_hPos[load_i][0];
            shared_pos_i[ty][1] = d_hPos[load_i][1];
            shared_pos_i[ty][2] = d_hPos[load_i][2];
        }
    }
    
    // Load tile of j-bodies into shared memory
    if (tx < TILE_SIZE) {
        int load_j = tile_j * TILE_SIZE + tx;
        if (load_j < numEntities) {
            shared_pos_j[tx][0] = d_hPos[load_j][0];
            shared_pos_j[tx][1] = d_hPos[load_j][1];
            shared_pos_j[tx][2] = d_hPos[load_j][2];
            shared_mass_j[tx] = d_mass[load_j];
        }
    }
    
    __syncthreads();
    
    // Compute acceleration using shared memory data
    // Since we use TILE_SIZE x TILE_SIZE blocks, ty and tx are always < TILE_SIZE
    // and i, j are always within the tile bounds for this block
    if (i < numEntities && j < numEntities) {
        if (i == j) {
            d_accels[i * numEntities + j][0] = 0.0;
            d_accels[i * numEntities + j][1] = 0.0;
            d_accels[i * numEntities + j][2] = 0.0;
        } else {
            // Use shared memory for positions and mass
            // Note: shared memory was loaded above, and we only use it if i,j < numEntities
            vector3 distance;
            distance[0] = shared_pos_i[ty][0] - shared_pos_j[tx][0];
            distance[1] = shared_pos_i[ty][1] - shared_pos_j[tx][1];
            distance[2] = shared_pos_i[ty][2] - shared_pos_j[tx][2];
            
            double magnitude_sq = distance[0] * distance[0] + 
                                  distance[1] * distance[1] + 
                                  distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            
            double accelmag = -1.0 * GRAV_CONSTANT * shared_mass_j[tx] / magnitude_sq;
            
            d_accels[i * numEntities + j][0] = accelmag * distance[0] / magnitude;
            d_accels[i * numEntities + j][1] = accelmag * distance[1] / magnitude;
            d_accels[i * numEntities + j][2] = accelmag * distance[2] / magnitude;
        }
    }
}

/*
 * Public API: compute() - same signature as serial version
 * Updates positions and velocities using CUDA kernels
 * Uses extern "C" to ensure C linkage for compatibility with nbody.c
 */
extern "C" void compute() {
    // Device array for accelerations (allocated once, reused each call)
    static vector3 *d_accels = NULL;
    static int allocated_size = 0;
    
    // Allocate acceleration matrix on first call
    if (d_accels == NULL || allocated_size != NUMENTITIES) {
        if (d_accels != NULL) {
            cudaFree(d_accels);
        }
        cudaError_t err = cudaMalloc((void**)&d_accels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);
        if (err != cudaSuccess) {
            #ifdef DEBUG
            fprintf(stderr, "CUDA malloc error for d_accels: %s\n", cudaGetErrorString(err));
            #endif
            return;
        }
        allocated_size = NUMENTITIES;
    }
    
    // Launch configuration for pairwise acceleration kernel
    // Use 2D grid with TILE_SIZE x TILE_SIZE blocks for shared memory optimization
    dim3 blockSize(TILE_SIZE, TILE_SIZE);  // TILE_SIZE x TILE_SIZE threads per block
    dim3 gridSize((NUMENTITIES + blockSize.x - 1) / blockSize.x,
                  (NUMENTITIES + blockSize.y - 1) / blockSize.y);
    
    // Kernel 1: Compute pairwise accelerations (using shared memory optimization)
    computePairwiseAccelerationsShared<<<gridSize, blockSize>>>(
        d_accels, d_hPos, d_mass, NUMENTITIES
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        #ifdef DEBUG
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        #endif
    }
    
    // Launch configuration for sum and update kernel
    // 1D grid: each thread handles one body
    dim3 blockSize1D(256);
    dim3 gridSize1D((NUMENTITIES + blockSize1D.x - 1) / blockSize1D.x);
    
    // Kernel 2: Sum accelerations and update velocities/positions
    sumAccelerationsAndUpdate<<<gridSize1D, blockSize1D>>>(
        d_accels, d_hVel, d_hPos, NUMENTITIES
    );
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        #ifdef DEBUG
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        #endif
    }
    
    // Synchronize to ensure kernels complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        #ifdef DEBUG
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        #endif
        // Don't return error here, but log it for debugging
    }
}

