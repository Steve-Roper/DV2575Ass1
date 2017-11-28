#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <device_functions.h>
#include <math.h>
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

__global__ void OddEvenSort(int *in_array, int *in_arraySize, const int *in_sorted, const int *in_sortAmount)
{
	bool sorted = true;
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2 * (*in_sortAmount);
	//swap evens iteration 1, swap odds iteration 2.
	for (int i = 0; i < *in_sortAmount; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			int tempindex = index + j;
			if (tempindex + 1 < *in_arraySize) //thread diversion in just one warp.
			{
				int min = tempindex + (int)(in_array[tempindex] > in_array[tempindex + 1]);
				int max = tempindex + (int)(in_array[tempindex] < in_array[tempindex + 1]);
				sorted = min <= max;

				min = in_array[min];
				max = in_array[max];

				in_array[tempindex] = min;
				in_array[tempindex + 1] = max;
				__syncthreads();
			}
		}
		index += 2;
	}
	//atomic multiplication (and) on in_notSorted to determmine if the array is sorted
	atomicMul((int*)in_sorted, (int)sorted);
}

int main()
{
	cudaError_t cudaStatus;

	dim3 grid_dim = dim3(/*how many blocks*/1, 1, 1);
	dim3 block_dim = dim3(/*how many threads*/1024 /*max per block for 900 series*/, 1, 1);

	const int arraySize = 14000;
	int *deviceArraySize = 0;
	cudaStatus = cudaMalloc(&deviceArraySize, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory for size of array\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy((void*)deviceArraySize, &arraySize, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not copy host arraySize to device\n");
		goto Error;
	}

	int* hostArray = (int*)malloc(arraySize * sizeof(int));
	srand(time(NULL));
	for (int i = 0; i < arraySize; ++i)
	{
		hostArray[i] = rand();
	}
	int* deviceArray;
	cudaStatus = cudaMalloc(&deviceArray, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory for array\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy((void*)deviceArray, (void*)hostArray, arraySize * (sizeof(int)), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not copy host array to device\n");
		goto Error;
	}

	int numZeros = 0;
	for (int i = 0; i < arraySize; ++i)
	{
		if (!hostArray[i])
			numZeros++;
	}
	printf("numZeros before: %d\n\n", numZeros);
	int sorted = 0;
	int *deviceSorted = 0;
	cudaStatus = cudaMalloc(&deviceSorted, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory for sorted int\n");
		goto Error;
	}
	int toSort = (arraySize / 2 - 1)/*how many threads we want to run*/ / (block_dim.x * grid_dim.x) /*how many threads we actually run*/ + 1;
	int *deviceToSort = 0;
	cudaStatus = cudaMalloc(&deviceToSort, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory for deviceToSort\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy((void*)deviceToSort, &toSort, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not copy host toSort to device\n");
		goto Error;
	}
	int calls = 0;
	timespec before;
	timespec_get(&before, TIME_UTC);

	while (!sorted)
	{
		calls++;
		sorted = 1;
		cudaStatus = cudaMemcpy((void*)deviceSorted, &sorted, sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			printf("Could not copy host sorted to device\n");
			goto Error;
		}
		OddEvenSort << <grid_dim, block_dim >> >(deviceArray, deviceArraySize, deviceSorted, deviceToSort);
		cudaDeviceSynchronize();

		cudaStatus = cudaMemcpy(&sorted, deviceSorted, sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			printf("Could not copy device sorted to host\n");
			goto Error;
		}
	}

	timespec after;
	timespec_get(&after, TIME_UTC);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to copy device array to host\n");
		goto Error;
	}
	numZeros = 0;
	for (int i = 0; i < arraySize; ++i)
	{
		printf("%d\t", hostArray[i]);
		if (!hostArray[i])
			numZeros++;
	}
	printf("\nnumZeros after: %d\n", numZeros);

	time_t timeTakenSec = after.tv_sec - before.tv_sec;
	long timeTakenNsec = after.tv_nsec - before.tv_nsec;
	timeTakenNsec = (timeTakenNsec < 0) ? -timeTakenNsec : timeTakenNsec;
	int timeTakenMsec = round(timeTakenNsec / 1000000.f);

	printf("\n%d kernel launch(es) over %lld seconds and %d milliseconds\n\n", calls, timeTakenSec, timeTakenMsec);
	
Error:
	free(hostArray);
	cudaFree(deviceArray);
	cudaFree(deviceArraySize);
	cudaFree(deviceSorted);
	cudaFree(deviceToSort);

	system("PAUSE");
	return 0;
}
