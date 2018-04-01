#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <device_functions.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <string>
#include <iostream>


#ifdef __INTELLISENSE__
void __syncthreads();
#endif

__global__ void KernelOddEvenSort(int *in_array, int *in_arraySize, int *in_stride, int* in_oddOrEven)
{
	int stride = *in_stride;
	int arraySize = *in_arraySize;
	int index = (threadIdx.x + blockIdx.x * blockDim.x) * 2 * stride;
	int compare[2];
	/*for(int i = 0; i < 2; ++i)
	{*/
		index += *in_oddOrEven;
		for (int j = 0; j < stride; ++j)
		{
			int tempindex = index + j * 2;
			if (tempindex + 1 < arraySize)
			{
				compare[0] = in_array[tempindex];
				compare[1] = in_array[tempindex + 1];
				int min = (int)(compare[0] > compare[1]);
				int max = (int)(compare[0] < compare[1]);
				min = compare[min];
				max = compare[max];
				in_array[tempindex] = min;
				in_array[tempindex + 1] = max;
				__syncthreads();
			}
		}
	//}
}

void OddEvenSort(int* in_array, const int *in_arraySize)
{
	for (int i = 0; i < *in_arraySize; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			for (int k = 0; k < *in_arraySize / 2; ++k)
			{
				int index = j + k * 2;
				if (index + 1 < *in_arraySize)
				{
					int min = index + (int)(in_array[index] > in_array[index + 1]);
					int max = index + (int)(in_array[index] < in_array[index + 1]);
					min = in_array[min];
					max = in_array[max];
					in_array[index] = min;
					in_array[index + 1] = max;
				}
			}
		}
	}
}

int main()
{
	cudaError_t cudaStatus;
	
	for (int i = 0; i < 10; ++i)
	{
		for (int j = 1000; j < 100001; j *= 10)
		{
			const int arraySize = j;
			int stride = 1;
			dim3 block_dim = dim3(/*how many threads*/128 / stride /*1024 max per block for 900 series*/, 1, 1);
			int blocks = (arraySize / 2 - 1)/*how many threads we want to run*/ / block_dim.x/*how many threads we actually run per block*/ + 1; //totals how many blocks to run
			dim3 grid_dim = dim3(blocks/*how many blocks*/, 1, 1);
			std::cout << "Started working on array of size " << arraySize << std::endl;
			/*Create host arrays, cpy is for CPU comparison*/
			int* hostArray = (int*)malloc(arraySize * sizeof(int));
			int* hostArraycpy = (int*)malloc(arraySize * sizeof(int));
			srand(time(NULL));
			for (int i = 0; i < arraySize; ++i)
			{
				hostArraycpy[i] = hostArray[i] = rand();
			}

			std::string filename;
			std::ofstream file;

			int *deviceStride = 0;
			cudaStatus = cudaMalloc(&deviceStride, sizeof(int));
			if (cudaStatus != cudaSuccess)
			{
				printf("Could not allocate device memory for deviceStride\n");
				goto Error;
			}
			cudaStatus = cudaMemcpy((void*)deviceStride, &stride, sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess)
			{
				printf("Could not copy host 'stride' to device\n");
				goto Error;
			}


			int *deviceArraySize = 0;
			cudaStatus = cudaMalloc(&deviceArraySize, sizeof(int));
			if (cudaStatus != cudaSuccess)
			{
				printf("Could not allocate device memory for deviceArraySize\n");
				goto Error;
			}
			cudaStatus = cudaMemcpy((void*)deviceArraySize, &arraySize, sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess)
			{
				printf("Could not copy host arraySize to device\n");
				goto Error;
			}


			int* deviceArray = 0;
			cudaStatus = cudaMalloc(&deviceArray, arraySize * sizeof(int));
			if (cudaStatus != cudaSuccess)
			{
				printf("Could not allocate device memory for array\n");
				goto Error;
			}
			cudaStatus = cudaMemcpy(deviceArray, hostArray, arraySize * sizeof(int), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess)
			{
				printf("Could not copy host array to device%d\n");
				goto Error;
			}

			//for debugging
			int occuring[10] = { 0 };
			//int numZeros = 0;
			for (int k = 0; k < arraySize; ++k)
			{
				//if (!hostArray[i])
					//numZeros++;
				if (hostArray[k] < 10)
					occuring[hostArray[k]]++;
			}
			//printf("Number of zeros before kernel sort: %d\n", numZeros);
			for (int k = 0; k < 10; ++k)
			{
				std::cout << "| " << k << " : " << occuring[k] << " ";
			}
			std::cout << std::endl;

			/*Kernel OddEven*/
			timespec before;
			timespec_get(&before, TIME_UTC);
			for (int k = 0; k < arraySize / 2; ++k)
			{
				for (int l = 0; l < 2; ++l)
				{
					int* deviceOddOrEven;
					cudaStatus = cudaMalloc(&deviceOddOrEven, sizeof(int));
					if (cudaStatus != cudaSuccess)
					{
						printf("Could not allocate device memory for odd/even\n");
						goto Error;
					}
					cudaStatus = cudaMemcpy(deviceOddOrEven, &l, sizeof(int), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess)
					{
						printf("Could not copy host odd/even to device%d\n");
						goto Error;
					}
					KernelOddEvenSort << <grid_dim, block_dim >> > (deviceArray, deviceArraySize, deviceStride, deviceOddOrEven);
				}
			}
			timespec after;
			timespec_get(&after, TIME_UTC);


			/*Copy output vector from GPU buffer to host memory.*/
			cudaStatus = cudaMemcpy(hostArray, deviceArray, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				printf("Failed to copy device array to host\n");
				goto Error;
			}
			std::fill(occuring, occuring + 10, 0);
			//numZeros = 0;
			bool notFailed = true;
			for (int k = 0; k < arraySize - 1; ++k)
			{
				//if (!hostArray[i])
					//++numZeros;
				if(hostArray[k] < 10)
					occuring[hostArray[k]]++;
				//printf("%d\t", hostArray[i]);
				if (hostArray[k] > hostArray[k + 1] && notFailed)
				{
					notFailed = false;
					printf("Kernel failed to sort at element: %d\n", k);
				}
			}
			if(hostArray[arraySize - 1] < 10)
				occuring[hostArray[arraySize - 1]]++;
			for (int k = 0; k < 10; ++k)
			{
				std::cout << "| " << k << " : " << occuring[k] << " ";
			}
			std::cout << std::endl;
			
			//printf("Number of zeros after kernel sort: %d\n", numZeros);

			time_t timeTakenSec = after.tv_sec - before.tv_sec;
			long timeTakenNsec = after.tv_nsec - before.tv_nsec;
			timeTakenNsec = (timeTakenNsec < 0) ? -timeTakenNsec : timeTakenNsec;
			int timeTakenMsec = round(timeTakenNsec / 1000000.f);

			printf("\nKernel sorted array of size %d in %lld seconds and %d milliseconds\n\n", arraySize, timeTakenSec, timeTakenMsec);

			filename = "C:/Users/Oskar/Desktop/tempData/DV2575Ass1_" + std::to_string(arraySize) + "_" + std::to_string(stride) + "_" + std::to_string(block_dim.x) + ".txt";
			file.open(filename, std::fstream::app);

			if (!file.fail())
			{
				file << i << '\t';
				file << timeTakenSec << '.';
				file << timeTakenMsec << '\n';
				file.close();
			}
			else
			{
				std::cerr << "Error: " << strerror(errno) << '\n';
			}

			/*CPU OddEven*/
			if (arraySize == 100000 && stride == 1 && block_dim.x == 128)
			{
				timespec_get(&before, TIME_UTC);
				OddEvenSort(hostArraycpy, &arraySize);
				timespec_get(&after, TIME_UTC);
				for (int i = 1; i < arraySize; ++i)
				{
					if (hostArraycpy[i - 1] > hostArraycpy[i])
					{
						printf("CPU failed to sort at element: %d\n", i);
						break;
					}
				}

				timeTakenSec = after.tv_sec - before.tv_sec;
				timeTakenNsec = after.tv_nsec - before.tv_nsec;
				timeTakenNsec = (timeTakenNsec < 0) ? -timeTakenNsec : timeTakenNsec;
				timeTakenMsec = round(timeTakenNsec / 1000000.f);

				printf("\nCPU sorted array of size %d in %lld seconds and %d milliseconds\n\n", arraySize, timeTakenSec, timeTakenMsec);

				filename = "C:/Users/Oskar/Desktop/tempData/DV2575Ass1_CPU_" + std::to_string(arraySize) + "_" + std::to_string(stride) + "_" + std::to_string(block_dim.x) + ".txt";
				file.open(filename, std::fstream::app);
				if (!file.fail())
				{
					file << i << '\t';
					file << timeTakenSec << '.';
					file << timeTakenMsec << '\n';
					file.close();
				}
				else
				{
					std::cerr << "Error: " << strerror(errno) << '\n';
				}
			}

		Error:
			free(hostArray);
			free(hostArraycpy);
			cudaFree(deviceArray);
			cudaFree(deviceArraySize);
		}
	}
	system("PAUSE");
	return 0;
}
