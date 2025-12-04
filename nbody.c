#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "planets.h"
#include "compute.h"

// represents the objects in the system.  Global variables
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass, *d_mass;

//initHostMemory: Create storage for numObjects entities in our system
//Parameters: numObjects: number of objects to allocate
//Returns: None
//Side Effects: Allocates memory in the hVel, hPos, and mass global variables
void initHostMemory(int numObjects)
{
	hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
	hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
	mass = (double *)malloc(sizeof(double) * numObjects);
}

//freeHostMemory: Free storage allocated by a previous call to initHostMemory
//Parameters: None
//Returns: None
//Side Effects: Frees the memory allocated to global variables hVel, hPos, and mass.
void freeHostMemory()
{
	free(hVel);
	free(hPos);
	free(mass);
}

//planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an estimation
//				of our solar system (Sun+NUMPLANETS)
//Parameters: None
//Returns: None
//Fills the first 8 entries of our system with an estimation of the sun plus our 8 planets.
void planetFill(){
	int i,j;
	double data[][7]={SUN,MERCURY,VENUS,EARTH,MARS,JUPITER,SATURN,URANUS,NEPTUNE};
	for (i=0;i<=NUMPLANETS;i++){
		for (j=0;j<3;j++){
			hPos[i][j]=data[i][j];
			hVel[i][j]=data[i][j+3];
		}
		mass[i]=data[i][6];
	}
}

//randomFill: FIll the rest of the objects in the system randomly starting at some entry in the list
//Parameters: 	start: The index of the first open entry in our system (after planetFill).
//				count: The number of random objects to put into our system
//Returns: None
//Side Effects: Fills count entries in our system starting at index start (0 based)
void randomFill(int start, int count)
{
	int i, j, c = start;
	for (i = start; i < start + count; i++)
	{
		for (j = 0; j < 3; j++)
		{
			hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
			hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
			mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
		}
	}
}

//printSystem: Prints out the entire system to the supplied file
//Parameters: 	handle: A handle to an open file with write access to prnt the data to
//Returns: 		none
//Side Effects: Modifies the file handle by writing to it.
void printSystem(FILE* handle){
	int i,j;
	for (i=0;i<NUMENTITIES;i++){
		fprintf(handle,"pos=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hPos[i][j]);
		}
		printf("),v=(");
		for (j=0;j<3;j++){
			fprintf(handle,"%lf,",hVel[i][j]);
		}
		fprintf(handle,"),m=%lf\n",mass[i]);
	}
}

int main(int argc, char **argv)
{
	clock_t t0=clock();
	int t_now;
	//srand(time(NULL));
	srand(1234);
	initHostMemory(NUMENTITIES);
	planetFill();
	randomFill(NUMPLANETS + 1, NUMASTEROIDS);
	
	/*
	 * CUDA Memory Management:
	 * - Allocate device arrays (d_hPos, d_hVel, d_mass) to mirror host arrays
	 * - Copy initial state from host to device before time loop
	 * - During time loop, compute() operates entirely on device arrays
	 * - After time loop, copy final state back to host for printing
	 */
	// Allocate device arrays for CUDA computation
	cudaError_t err;
	err = cudaMalloc((void**)&d_hPos, sizeof(vector3) * NUMENTITIES);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA malloc error for d_hPos: %s\n", cudaGetErrorString(err));
		return 1;
	}
	err = cudaMalloc((void**)&d_hVel, sizeof(vector3) * NUMENTITIES);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA malloc error for d_hVel: %s\n", cudaGetErrorString(err));
		cudaFree(d_hPos);
		return 1;
	}
	err = cudaMalloc((void**)&d_mass, sizeof(double) * NUMENTITIES);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA malloc error for d_mass: %s\n", cudaGetErrorString(err));
		cudaFree(d_hPos);
		cudaFree(d_hVel);
		return 1;
	}
	
	// Copy initial state from host to device
	err = cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA memcpy error (H2D d_hPos): %s\n", cudaGetErrorString(err));
		cudaFree(d_hPos);
		cudaFree(d_hVel);
		cudaFree(d_mass);
		return 1;
	}
	err = cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA memcpy error (H2D d_hVel): %s\n", cudaGetErrorString(err));
		cudaFree(d_hPos);
		cudaFree(d_hVel);
		cudaFree(d_mass);
		return 1;
	}
	err = cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA memcpy error (H2D d_mass): %s\n", cudaGetErrorString(err));
		cudaFree(d_hPos);
		cudaFree(d_hVel);
		cudaFree(d_mass);
		return 1;
	}
	
	//now we have a system.
	#ifdef DEBUG
	printSystem(stdout);
	#endif
	for (t_now=0;t_now<DURATION;t_now+=INTERVAL){
		compute();
	}
	clock_t t1=clock()-t0;
	
	// Copy final state from device back to host (for printSystem)
	err = cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA memcpy error (D2H hPos): %s\n", cudaGetErrorString(err));
	}
	err = cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA memcpy error (D2H hVel): %s\n", cudaGetErrorString(err));
	}
	
#ifdef DEBUG
	printSystem(stdout);
#endif
	printf("This took a total time of %f seconds\n",(double)t1/CLOCKS_PER_SEC);

	// Free device arrays
	err = cudaFree(d_hPos);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA free error for d_hPos: %s\n", cudaGetErrorString(err));
	}
	err = cudaFree(d_hVel);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA free error for d_hVel: %s\n", cudaGetErrorString(err));
	}
	err = cudaFree(d_mass);
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA free error for d_mass: %s\n", cudaGetErrorString(err));
	}
	
	freeHostMemory();
	return 0;
}
