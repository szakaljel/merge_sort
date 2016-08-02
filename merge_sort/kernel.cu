
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;


__device__ void merge(int * start_a,int * start_tmp,int count)
{
	int elementsCompared = 0;
	int *current_first = start_a;
	int *last_first = current_first + (count / 2);
	int *current_second = start_a + (count / 2);
	int *last_second = current_second + (count / 2);
	int tmp;

	for (int i = 0; i < count; i++)
	{
		if (current_first < last_first)
		{
			if (current_second < last_second)
			{
				if (current_first[0] <= current_second[0])
				{
					tmp = current_first[0];
					start_tmp[i] = tmp;
					current_first += 1;
				}
				else
				{
					tmp = current_second[0];
					start_tmp[i] = tmp;
					current_second += 1;
				}
			}
			else
			{
				tmp = current_first[0];
				start_tmp[i] = tmp;
				current_first += 1;
			}
		}
		else
		{
			tmp = current_second[0];
			start_tmp[i] = tmp;
			current_second += 1;
		}
	}

	for (int i = 0; i < count; i++)
	{
		start_a[i] = start_tmp[i];
	}
}

__global__ void kernel(int * da, int * d_tmp)
{
	int tid = threadIdx.x;
	int activeThreads = blockDim.x;
	int jump = 2;
	int *start_a;
	int *start_tmp;

	while (jump <= 2*blockDim.x)
	{
		if (tid < activeThreads)
		{
			start_a = da + jump*tid;
			start_tmp = d_tmp + jump*tid;
			merge(start_a, start_tmp, jump);
			__syncthreads();
			
		}
		activeThreads = activeThreads/2;
		jump = jump*2;

	}

}


int compare(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

int main()
{

	int ElementCount = 1024;
	int TotalSize = ElementCount * sizeof(int);
	int *table;

	cudaError_t error;
	cudaSetDevice(0);
	cudaSetDevice(cudaDeviceMapHost);

	error = cudaHostAlloc(&table, TotalSize, cudaHostAllocMapped);
	
	srand(time(0));

	for (int i = 0; i < ElementCount; i++)
	{
		table[i] = rand() % 1000;
	}
	

	int * da, *d_tmp; 
	error = cudaHostGetDevicePointer(&da, table, 0);
	error = cudaMalloc(&d_tmp, TotalSize);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernel<<<1, ElementCount / 2 >>> (da, d_tmp);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float duration;

	cudaEventElapsedTime(&duration, start, stop);

	cout << duration << " ms" << endl;

	for (int i = 0; i < ElementCount; i++)
	{
		if (i % 8 == 0)
		{
			cout << endl;
		}
		cout << table[i] << "  ";
		
	}
	
	cudaFree(da);
	cudaFreeHost(table);
	

	getchar();

	/*table = new int[ElementCount];

	for (int i = 0; i < ElementCount; i++)
	{
		table[i] = rand() % 1000;
	}


	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	qsort(table, ElementCount, sizeof(int), compare);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration2 = duration_cast<milliseconds>(t2 - t1).count();

	cout << duration2 <<" ms"<<endl;
	*/
	/*for (int i = 0; i < ElementCount; i++)
	{
		if (i % 8 == 0)
		{
			cout << endl;
		}
		cout << table[i] << "  ";

	}*/
	

/*	delete[] table;

	getchar();
	*/
    return 0;
}
