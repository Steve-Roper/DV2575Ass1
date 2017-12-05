#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <device_functions.h>
#include <math.h>
#include <time.h>

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

__global__ void OddEvenSort(int *in_array, int *in_arraySize, int *in_sortAmount, int *in_threads)
{
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2;//* *in_sortAmount;
	int compare[2];
	int threads = *in_threads;
	for(int i = 0; i < 2; ++i)
	{
		index += i;
		for(int j = 0; j < *in_sortAmount; ++j)
		{
			int tempindex = index + j * 2 * threads;
			if(tempindex + 1 < *in_arraySize)
			{
				compare[0] = in_array[tempindex];
				compare[1] = in_array[tempindex + 1];
				int min = (int)(compare[0] > compare[1]);
				int max = (int)(compare[0] < compare[1]);
				/*if (!index && j == 1)
				{
					printf("From compare:\nmin pos: %d,\tmax pos: %d\n", min, max);
				}*/
				min = compare[min];
				max = compare[max];
				/*if (!index && j == 1)
				{
					printf("min val: %d,\tmax val: %d\n", min, max);
					printf("      0: %d,\t      1: %d\n", compare[0], compare[1]);
				}*/

				/*min = tempindex + (int)(in_array[tempindex] > in_array[tempindex + 1]);
				max = tempindex + (int)(in_array[tempindex] < in_array[tempindex + 1]);
				if (!index && !j)
				{
					printf("From array:\nmin pos: %d,\tmax pos: %d\n", min, max);
				}
				min = in_array[min];
				max = in_array[max];
				if (!index && !j)
				{
					printf("min val: %d,\tmax val: %d\n", min, max);
				}
*/
				in_array[tempindex] = min;
				in_array[tempindex + 1] = max;
				__syncthreads();
			}
		}
	}
}

int main()
{
	cudaError_t cudaStatus;

	dim3 grid_dim = dim3(/*how many blocks*/9, 1, 1); //940m has 384 threads total
	dim3 block_dim = dim3(/*how many threads*/1024 /*1024 max per block for 900 series*/, 1, 1);
	int noThreads = grid_dim.x * block_dim.x;
	int *deviceNoThreads = 0;
	cudaStatus = cudaMalloc(&deviceNoThreads, sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not allocate device memory for size of array\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy((void*)deviceNoThreads, &noThreads, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not copy host arraySize to device\n");
		goto Error;
	}
	const int arraySize = 140000;
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

	int numZeros = 0;
	for (int i = 0; i < arraySize; ++i)
	{
		if (!hostArray[i])
			numZeros++;
	}
	printf("numZeros before: %d\n\n", numZeros);

	int toSort = (arraySize / 2 - 1)/*how many threads we want to run*/ / noThreads /*how many threads we actually run*/ + 1; //how many odd+even each thread will perform
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
	cudaStatus = cudaMemcpy(deviceArray, hostArray, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		printf("Could not copy host array to device%d\n");
		goto Error;
	}
	timespec before;
	timespec_get(&before, TIME_UTC);

	for(int i = 0; i < arraySize / 2; ++i)
	{
		OddEvenSort << <grid_dim, block_dim >> >(deviceArray, deviceArraySize, deviceToSort, deviceNoThreads);
		//cudaDeviceSynchronize();
	}

	timespec after;
	timespec_get(&after, TIME_UTC);
	cudaStatus = cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to copy device array to host%d\n");
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to copy device array to host\n");
		goto Error;
	}
	numZeros = 0;
	int failed = 0;
	printf("%d\t", hostArray[0]);
	if (!hostArray[0])
		numZeros++;
	for (int i = 1; i < arraySize; ++i)
	{
		printf("%d\t", hostArray[i]);
		if (!hostArray[i])
			numZeros++;
		if (hostArray[i - 1] > hostArray[i])
		{
			failed = i;
			break;
		}
	}
	printf("\nnumZeros after: %d\n", numZeros);

	time_t timeTakenSec = after.tv_sec - before.tv_sec;
	long timeTakenNsec = after.tv_nsec - before.tv_nsec;
	timeTakenNsec = (timeTakenNsec < 0) ? -timeTakenNsec : timeTakenNsec;
	int timeTakenMsec = round(timeTakenNsec / 1000000.f);

	printf("\nSorted in %lld seconds and %d milliseconds\n\n", timeTakenSec, timeTakenMsec);

	if (failed)
		printf("\Failed to sort array at element %d\n\n", failed);

Error:
	free(hostArray);
	cudaFree(deviceArray);
	cudaFree(deviceArraySize);
	cudaFree(deviceToSort);

	system("PAUSE");
	return 0;
}
