
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;


#define GLOBAL_ELEMENT_SIZE  1024

__device__ void merge(int * start_a,int * start_tmp,int count)
{
	int first_x;
	int second_x;
	int *current_first = start_a;
	int *last_first = current_first + (count / 2);
	int *current_second = start_a + (count / 2);
	int *last_second = current_second + (count / 2);



	for (int i = 0; i < count; i++)
	{
		second_x = current_second < last_second ? current_second[0] : 0x7FFFFFFF;
		first_x = current_first < last_first ? current_first[0] : 0x7FFFFFFF;

		start_tmp[i] = second_x < first_x ? second_x : first_x;
		current_second = second_x < first_x ? current_second+1 : current_second;
		current_first = second_x < first_x ? current_first : current_first+1;

	}

	for (int i = 0; i < count; i++)
	{
		start_a[i] = start_tmp[i];
	}

	return;
}

__global__ void kernel(int * da)
{
	__shared__ int tmp_memory[GLOBAL_ELEMENT_SIZE];
	__shared__ int swap_memory[GLOBAL_ELEMENT_SIZE];
	
	int tid = threadIdx.x;
	int activeThreads = blockDim.x;
	int jump = 2;
	int *start_a;
	int *start_tmp;

	if (tid == 0)
	{
		for (int i = 0; i < GLOBAL_ELEMENT_SIZE; i++)
		{
			tmp_memory[i] = da[i];
		}
	}
	__syncthreads();
	while (jump <= 2*blockDim.x)
	{
		if (tid < activeThreads)
		{
			start_a = tmp_memory + jump*tid;
			start_tmp = swap_memory + jump*tid;
			merge(start_a, start_tmp, jump);
			__syncthreads();
			
		}
		activeThreads = activeThreads/2;
		jump = jump*2;

	}
	if (tid == 0)
	{
		for (int i = 0; i < GLOBAL_ELEMENT_SIZE; i++)
		{
			da[i] = tmp_memory[i];
		}
	}

}


int compare(const void * a, const void * b)
{
	return (*(int*)a - *(int*)b);
}

int main()
{

	int ElementCount = GLOBAL_ELEMENT_SIZE;
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
	

	int * da; 
	error = cudaHostGetDevicePointer(&da, table, 0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	kernel<<<1, ElementCount / 2 ,TotalSize*2>>> (da);

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
