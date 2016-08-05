
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;


#define CHUNK_ELEMENT_COUNT  2048
#define FULL_ELEMENT_COUNT 128 * CHUNK_ELEMENT_COUNT

__device__ void merge(int * start_a,int * start_tmp,int count)
{

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
				if (current_first[0] < current_second[0])
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

__global__ void kernel(int * da)
{
	__shared__ int tmp_memory[CHUNK_ELEMENT_COUNT];
	__shared__ int swap_memory[CHUNK_ELEMENT_COUNT];
	
	int tid = threadIdx.x;
	int offset = blockIdx.x * CHUNK_ELEMENT_COUNT;
	int activeThreads = blockDim.x;
	int jump = 2;
	int *start_a;
	int *start_tmp;

	tmp_memory[2 * tid] = da[2 * tid+offset];
	tmp_memory[2 * tid + 1] = da[2 * tid + 1+offset];

	__syncthreads();
	while (jump <= 2*blockDim.x)
	{
		if (tid < activeThreads)
		{
			start_a = tmp_memory + jump*tid;
			start_tmp = swap_memory + jump*tid;
			merge(start_a, start_tmp, jump);
			
			
		}
		activeThreads = activeThreads/2;
		jump = jump*2;
		__syncthreads();

	}
	
	da[2 * tid + offset] = tmp_memory[2 * tid];
	da[2 * tid + 1 + offset] = tmp_memory[2 * tid + 1];


}

__global__ void kernel_second_merge(int * da, int* dtmp)
{
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int activeThreads = gridDim.x * blockDim.x;
	int jump = CHUNK_ELEMENT_COUNT*2;
	int *start_a;
	int *start_tmp;
	while (jump <= FULL_ELEMENT_COUNT)
	{
		if (tid < activeThreads)
		{
			start_a = da + jump*tid;
			start_tmp = dtmp + jump*tid;
			merge(start_a, start_tmp, jump);
			
			
		}
		activeThreads = activeThreads/2;
		jump = jump*2;
		__syncthreads();

	}

}

int compare(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}


bool is_sort(int* tab,int count)
{
	for(int i=0;i<count-1;i++)
	{
		if(tab[i]>tab[i+1])
		{
			cout<<i<<endl;
			return false;
		}
	}
	return true;
}


int main()
{

	int ElementCount = FULL_ELEMENT_COUNT;
	int ChunkCount = CHUNK_ELEMENT_COUNT;
	int FullSize = ElementCount * sizeof(int);
	int ChunkSize = ChunkCount * sizeof(int);
	int *table;
	int * result;

	cudaError_t error;
	cudaSetDevice(0);
	cudaSetDevice(cudaDeviceMapHost);

	error = cudaHostAlloc(&table, FullSize, cudaHostAllocMapped);
	error = cudaHostAlloc(&result, FullSize, cudaHostAllocMapped);

	srand(time(0));

	for (int i = 0; i < ElementCount; i++)
	{
		table[i] = rand() % 1000000;
	}
	


	int * da; 
	error = cudaHostGetDevicePointer(&da, table, 0);
	
	int * dtmp;
	error = cudaHostGetDevicePointer(&dtmp, result, 0);

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	//cudaEventRecord(start);

	high_resolution_clock::time_point tt1 = high_resolution_clock::now();
	kernel<<<ElementCount/ChunkCount, ChunkCount / 2 ,ChunkSize*2>>> (da);
	kernel_second_merge<<<1,(ElementCount/ChunkCount)/2>>>(da,dtmp);
	
	cudaThreadSynchronize();
	high_resolution_clock::time_point tt2 = high_resolution_clock::now();

	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);

	//float duration;
	auto duration = duration_cast<milliseconds>(tt2 - tt1).count();

	//cudaEventElapsedTime(&duration, start, stop);
	

	cout << duration << " ms" << "  sort: "<<is_sort(table,ElementCount)<< endl;

	/*for (int i = 0; i < 8096; i++)
	{
		if (i % 16 == 0)
		{
			cout << endl;
		}
		cout << table[i] << "  ";
		
	}*/
	
	cudaFree(da);
	cudaFreeHost(table);

	cudaFree(dtmp);
	cudaFreeHost(result);
	

	getchar();

	table = new int[ElementCount];

	for (int i = 0; i < ElementCount; i++)
	{
		table[i] = rand() % 1000000;
	}


	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	qsort(table, ElementCount, sizeof(int), compare);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	auto duration2 = duration_cast<milliseconds>(t2 - t1).count();

	cout << duration2 << " ms" << "  sort: "<<is_sort(table,ElementCount)<< endl;

	/*for (int i = 0; i < ElementCount; i++)
	{
		if (i % 8 == 0)
		{
			cout << endl;
		}
		cout << table[i] << "  ";

	}*/
	

	delete[] table;

	getchar();
	
    return 0;
}
