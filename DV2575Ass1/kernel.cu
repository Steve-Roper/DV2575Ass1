#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <device_functions.h>

#include <time.h>

#ifdef __INTELLISENSE__
void __syncthreads();
int atomicCAS(unsigned long long int* old, unsigned long long int comp, unsigned long long int val);
#endif

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__ int atomicMul(int* address, int val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, (unsigned long long int)(val * assumed));

	} while (assumed != old);

	return old;
}

__global__ void OddEvenSort(int *in_array, const int *in_arraySize)
{
	bool sorted = true;
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2;

	//swap evens iteration 1, swap odds iteration 2.
	for (int i = 0; i < *in_arraySize - 1; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			int min = in_array[index + (in_array[index] > in_array[index + 1])];
			int max = in_array[index + (in_array[index] <= in_array[index + 1])];

			in_array[index] = min;
			in_array[index++] = max;
			sorted = false;
			__syncthreads();
		}
	}

	//atomic multiplication on in_notSorted to determmine if the array is sorted
	
	//atomicMul((int*)in_notSorted, (int)sorted);

}

int main()
{
	const int arraySize = 2048;//32 * 4 * 13 * 2; //max number of threads, times 2 b/c 2 are compared
	cudaError_t cudaStatus;
	int* hostArray = (int*)malloc(arraySize * sizeof(int));
	int* deviceArray;
	cudaStatus = cudaMalloc(&deviceArray, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory\n");
		goto Error;
	}

	srand(time(NULL));
	for (int i = 0; i < arraySize; ++i)
	{
		hostArray[i] = rand();
	}

	cudaStatus = cudaMemcpy(deviceArray, hostArray, arraySize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not copy host memory to device\n");
		goto Error;
	}

	dim3 block_dim = dim3(/*how many threads*/1024 /*max for 900 series*/, 1, 1);
	dim3 grid_dim = dim3(/*how many blocks*/1, 1, 1);

	//bool notSorted = true;
	//while (notSorted)
	//{
		OddEvenSort <<<grid_dim, block_dim >>>(hostArray, &arraySize);
	//}
	/*const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addWithCuda failed!");
	return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceReset failed!");
	return 1;
	}*/
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to copy from device to host\n");
		goto Error;
	}

	for (int i = 0; i < arraySize; ++i)
	{
		printf("%d\t", hostArray[i]);
	}
	
Error:
	free(hostArray);
	cudaFree(deviceArray);

	system("PAUSE");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
