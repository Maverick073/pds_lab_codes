# Overview of CUDA

**CUDA (Compute Unified Device Architecture)** is a parallel computing platform and programming model developed by **NVIDIA**. It allows developers to use GPUs (Graphics Processing Units) for general-purpose processing, an approach known as **GPGPU** (General-Purpose computing on Graphics Processing Units).

CUDA provides a powerful framework that enables developers to accelerate their applications by offloading compute-intensive tasks to the GPU. GPUs are highly parallel devices, which makes them ideal for tasks like scientific simulations, deep learning, and image processing.

## Why CUDA is Used

### 1. **Parallel Computing**
   - CUDA takes advantage of the GPU’s multiple cores, making it well-suited for applications that can be parallelized.
   - It can perform many operations simultaneously, speeding up computations drastically compared to traditional CPU-only processing.

### 2. **High Performance**
   - CUDA can provide significant performance improvements for certain applications, especially those that require high levels of computation, such as simulations, video rendering, and machine learning.

### 3. **Ease of Use**
   - CUDA uses a C/C++ based programming model, so developers familiar with these languages can easily adapt to writing GPU-accelerated code.
   - It provides libraries, tools, and compiler support to make development easier.

### 4. **Accelerated Libraries**
   - CUDA offers several optimized libraries for specific tasks such as deep learning (cuDNN), linear algebra (cuBLAS), and scientific computing (cuFFT).
   
### 5. **Scalability**
   - CUDA allows programs to scale across multiple GPUs, making it ideal for complex, large-scale computations and data processing.

## Common Applications of CUDA
- **Deep Learning**: Used in training machine learning models, especially in frameworks like TensorFlow and PyTorch.
- **Image & Video Processing**: Used for real-time video editing, rendering, and computer vision tasks.
- **Scientific Simulations**: Simulations of physical, chemical, or biological systems that require significant computational power.
- **Cryptography**: Performing fast encryption and decryption operations, as well as blockchain mining.

CUDA has revolutionized many industries by enabling faster processing and reducing computational costs for complex operations.

# CUDA Function Signatures with Examples

Below is a table summarizing commonly used CUDA functions with their respective signatures, descriptions, and examples.

| **Function Name**          | **Signature**                                                                               | **Description**                                                                                  | **Example**                                                                                                                                       |
|----------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `cudaMalloc`               | `cudaError_t cudaMalloc(void** devPtr, size_t size)`                                        | Allocates memory on the GPU device.                                                             | ```int* d_array; cudaMalloc((void**)&d_array, 100 * sizeof(int));```                                                                   |
| `cudaFree`                 | `cudaError_t cudaFree(void* devPtr)`                                                       | Frees previously allocated GPU memory.                                                          | ```cudaFree(d_array);```                                                                                                                    |
| `cudaMemcpy`               | `cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)`     | Copies memory between host and device.                                                          | ```cudaMemcpy(d_array, h_array, 100 * sizeof(int), cudaMemcpyHostToDevice);```                                                             |
| `cudaMemcpyToSymbol`       | `cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, cudaMemcpyKind kind)` | Copies data to constant memory on the device.                                                   | ```cudaMemcpyToSymbol(d_radius, &radius, sizeof(int));```                                                                                  |
| `cudaEventCreate`          | `cudaError_t cudaEventCreate(cudaEvent_t* event)`                                          | Creates an event for timing or synchronization.                                                 | ```cudaEvent_t start; cudaEventCreate(&start);```                                                                                        |
| `cudaEventRecord`          | `cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0)`                    | Records an event in a specific stream.                                                          | ```cudaEventRecord(start);```                                                                                                               |
| `cudaEventSynchronize`     | `cudaError_t cudaEventSynchronize(cudaEvent_t event)`                                      | Waits for an event to complete.                                                                 | ```cudaEventSynchronize(stop);```                                                                                                          |
| `cudaEventElapsedTime`     | `cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop)`         | Computes elapsed time in milliseconds between two events.                                        | ```float time; cudaEventElapsedTime(&time, start, stop);```                                                                              |
| `cudaEventDestroy`         | `cudaError_t cudaEventDestroy(cudaEvent_t event)`                                          | Destroys a previously created event.                                                            | ```cudaEventDestroy(start);```                                                                                                              |
| `cudaDeviceSynchronize`    | `cudaError_t cudaDeviceSynchronize()`                                                      | Blocks the host until all preceding device work is complete.                                    | ```cudaDeviceSynchronize();```                                                                                                              |
| `cudaMemcpyKind`           | `enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost }` | Specifies direction of memory copy operation.                                                   | ```cudaMemcpy(d_array, h_array, 100 * sizeof(int), cudaMemcpyHostToDevice);```                                                             |
| `dim3`                     | `dim3(unsigned int x=1, unsigned int y=1, unsigned int z=1)`                               | Defines grid/block dimensions for kernel launch.                                                | ```dim3 grid(16, 16); dim3 block(16, 16);```                                                                                             |
| `__syncthreads`            | `__device__ void __syncthreads()`                                                          | Synchronizes threads in a block.                                                                | ```__syncthreads();```                                                                                                                      |
| `cudaGetLastError`         | `cudaError_t cudaGetLastError()`                                                           | Returns the last error from a CUDA runtime call.                                                | ```cudaError_t err = cudaGetLastError();```                                                                                                 |
| `cudaMemcpyAsync`          | `cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0)` | Asynchronously copies memory between host and device.                                            | ```cudaMemcpyAsync(d_array, h_array, 100 * sizeof(int), cudaMemcpyHostToDevice, stream);```                                                 |
| `cudaStreamCreate`         | `cudaError_t cudaStreamCreate(cudaStream_t* pStream)`                                      | Creates a new stream for asynchronous operations.                                               | ```cudaStream_t stream; cudaStreamCreate(&stream);```                                                                                    |
| `cudaStreamDestroy`        | `cudaError_t cudaStreamDestroy(cudaStream_t stream)`                                       | Destroys a previously created stream.                                                           | ```cudaStreamDestroy(stream);```                                                                                                            |
| `cudaStreamSynchronize`    | `cudaError_t cudaStreamSynchronize(cudaStream_t stream)`                                   | Blocks the host until all operations in the stream are complete.                                | ```cudaStreamSynchronize(stream);```                                                                                                        |
| `cudaLaunchKernel`         | `cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)` | Launches a kernel on the device.                                                                | ```cudaLaunchKernel((void*)&kernelFunc, grid, block, args, 0, stream);```                                                                   |
| `cudaOccupancyMaxPotentialBlockSize` | `cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize=0, int blockSizeLimit=0)` | Determines optimal block size and grid size for a kernel launch.                                 | ```int minGridSize, blockSize; cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernelFunc, 0, 0);```                         |


## Steps to Compile and Run a CUDA Program

### 1. **Write the CUDA Program**

Create a CUDA program using any text editor (e.g., Visual Studio Code, Sublime Text, or Nano). Ensure that the file extension is `.cu`. Example:

```C
#include <stdio.h>

__global__ void helloCUDA()
{
    printf("Hello, CUDA World!\n");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 2. Compilation and Execution of Program 

- Save the file with a `.cu` extension (e.g., `hello_cuda.cu`).
- Compile the program using the `nvcc` compiler: `nvcc <filename.cu>`
- Run `./a.out`


## EXP 1 - DOT PRODUCT OF TWO VECTORS

```C
#include <stdio.h>
#include <cuda.h>

__global__ void product(int* a, int* b, int* c, int size) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t < size) {
        c[t] = a[t] * b[t];
    }
}

__global__ void sum(int* c, int* partial_sums, int size) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        sdata[tid] = c[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

int main() {
    int *a, *b, *c, i, N;
    int size;
    
    printf("Enter the size of the arrays: ");
    scanf("%d", &N);
    size = N * sizeof(int);
    
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    
    int ans = 0;
    
    int *d_a, *d_b, *d_c, *d_partial_sums;
    
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_partial_sums, sizeof(int) * ((N + 255) / 256));
    
    printf("Enter array a:\n");
    for (i = 0; i < N; i++) {
        scanf("%d", &a[i]);
    }
    
    printf("Enter array b:\n");
    for (i = 0; i < N; i++) {
        scanf("%d", &b[i]);
    }
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    product<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    sum<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_c, d_partial_sums, N);

    int* partial_sums = (int*)malloc(sizeof(int) * blocksPerGrid);
    cudaMemcpy(partial_sums, d_partial_sums, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost);
    
    ans = 0;
    for (i = 0; i < blocksPerGrid; i++) {
        ans += partial_sums[i];
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("DOT PRODUCT = %d\n", ans);
    printf("Elapsed time: %f ms\n", milliseconds);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_partial_sums);
    
    free(a);
    free(b);
    free(c);
    free(partial_sums);
    
    return 0;
}

```
### Explanation of Output:

- The program first prompts the user to enter the size of the arrays. In this case, we enter `5` for the size of the arrays.
- Then, the program asks the user to enter the elements of two arrays (`a` and `b`).
- For the input arrays:
  - Array `a = [1, 2, 3, 4, 5]`
  - Array `b = [6, 7, 8, 9, 10]`
  
  The dot product is calculated as:

  \[
  1* 6 + 2* 7 + 3* 8 + 4* 9 + 5* 10 = 6 + 14 + 24 + 36 + 50 = 130
  \]

- Finally, the program prints the calculated **dot product** and the **elapsed time** for the GPU computation. The time may vary depending on the hardware and environment.



## EXP 2 - MATRIX TRANSPOSE USING SHARED MEMORY

```cpp
#include <stdio.h>
#include <cuda.h>

__global__ void mtrans_smem(int *a, int *b, int rows, int cols) {
    __shared__ int tile[16][16];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int from = y * cols + x;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = a[from];
    }
    __syncthreads();

    int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    int irow = bidx / blockDim.y;
    int icol = bidx % blockDim.y;
    x = blockIdx.y * blockDim.y + icol;
    y = blockIdx.x * blockDim.x + irow;
    int to = y * rows + x;

    if (x < rows && y < cols) {
        b[to] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int rows, cols;
    printf("Enter number of rows: ");
    scanf("%d", &rows);
    printf("Enter number of columns: ");
    scanf("%d", &cols);

    int *a, *b;
    int size = rows * cols * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);

    int *d_a, *d_b;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);

    printf("Enter the elements of the matrix:\n");
    for (int i = 0; i < rows * cols; i++) {
        scanf("%d", &a[i]);
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mtrans_smem<<<grid, block>>>(d_a, d_b, rows, cols);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    printf("Transposed matrix:\n");
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("\t%d", b[i * rows + j]);
        }
        printf("\n");
    }

    printf("Elapsed time : %f ms \n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);
    cudaDeviceReset();

    return 0;
}
```

### Exp 3 - 1D Stencil Using Constant Memory
```cpp
#include <stdio.h>
#include <cuda.h>

__constant__ int d_radius;

__global__ void stencilKernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int sum = 0;
        for (int offset = -d_radius; offset <= d_radius; ++offset) {
            int neighborIdx = idx + offset;
            if (neighborIdx >= 0 && neighborIdx < n) {
                sum += input[neighborIdx];
            }
        }
        output[idx] = sum;
    }
}

int main() {
    int n, radius;

    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_output = (int*)malloc(n * sizeof(int));

    printf("Enter the elements of the array:\n");
    for (int i = 0; i < n; ++i) {
        scanf("%d", &h_input[i]);
    }

    printf("Enter the radius: ");
    scanf("%d", &radius);

    int *d_input, *d_output;
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_radius, &radius, sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    stencilKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    printf("Elapsed time : %f ms\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}

### Experiment 4 - Pre-Order Tree Traversal
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 1024 // Maximum number of nodes
#define THREADS_PER_BLOCK 32

__global__ void preorder_traversal_kernel(int *tree, int *output, int *output_idx, int num_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_nodes && tree[idx] != -1) {
        int stack[N];
        int top = -1;

        // Initialize stack with root node if it's the first thread in the block
        if (threadIdx.x == 0) {
            stack[++top] = idx;
        }

        while (top >= 0) {
            int current_node = stack[top--];
            int output_position = atomicAdd(output_idx, 1);

            // Output the current node
            output[output_position] = tree[current_node];

            // Push right and then left child to stack
            int right_child = 2 * current_node + 2;
            int left_child = 2 * current_node + 1;

            if (right_child < num_nodes && tree[right_child] != -1) {
                stack[++top] = right_child;
            }
            if (left_child < num_nodes && tree[left_child] != -1) {
                stack[++top] = left_child;
            }
        }
    }
}

void preorder_traversal_cuda(int *tree, int *output, int num_nodes) {
    int *d_tree, *d_output, *d_output_idx;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_tree, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_output, num_nodes * sizeof(int));
    cudaMalloc((void**)&d_output_idx, sizeof(int));

    // Copy tree to GPU
    cudaMemcpy(d_tree, tree, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize output array on GPU
    cudaMemset(d_output, -1, num_nodes * sizeof(int));

    // Initialize output index on GPU
    int initial_idx = 0;
    cudaMemcpy(d_output_idx, &initial_idx, sizeof(int), cudaMemcpyHostToDevice);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel with multiple threads
    int blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    preorder_traversal_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_tree, d_output, d_output_idx, num_nodes);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(output, d_output, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_tree);
    cudaFree(d_output);
    cudaFree(d_output_idx);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Output elapsed time
    std::cout << "Time for the kernel execution: " << elapsedTime << " ms" << std::endl;
}

int main() {
    int num_nodes;

    std::cout << "Enter the number of nodes in the binary tree: ";
    std::cin >> num_nodes;

    int *tree = new int[num_nodes];
    std::cout << "Enter the elements of the binary tree in level-order (use -1 for null nodes):" << std::endl;
    for (int i = 0; i < num_nodes; i++) {
        std::cin >> tree[i];
    }

    int *output = new int[num_nodes];
    preorder_traversal_cuda(tree, output, num_nodes);

    std::cout << "Preorder Traversal: ";
    for (int i = 0; i < num_nodes; i++) {
        if (output[i] != -1)
            std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    delete[] tree;
    delete[] output;

    return 0;
}

### Experiment 5 - Odd-Even Transposition Sort
```cpp
#include <stdio.h>
#include <cuda.h>

__device__ int custom_max(int a, int b) {
    return a > b ? a : b;
}

__device__ int custom_min(int a, int b) {
    return a < b ? a : b;
}

__global__ void sort(int* a, int n) {
    int tid = threadIdx.x;

    for (int i = 0; i < n; i++) {
        // Odd phase
        if ((tid % 2 == 1) && (tid < n - 1)) {
            int t = a[tid + 1];
            a[tid + 1] = custom_max(a[tid], t);
            a[tid] = custom_min(a[tid], t);
        }
        __syncthreads();

        // Even phase
        if ((tid % 2 == 0) && (tid < n - 1)) {
            int t = a[tid + 1];
            a[tid + 1] = custom_max(a[tid], t);
            a[tid] = custom_min(a[tid], t);
        }
        __syncthreads();
    }
}

int main() {
    int n;

    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *a = (int *)malloc(n * sizeof(int));
    printf("Enter %d elements: ", n);
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }

    printf("ARRAY ELEMENTS BEFORE Sorting:");
    for (int i = 0; i < n; i++) {
        printf(" %d", a[i]);
    }
    printf("\n");

    int* d_a;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sort<<<1, n>>>(d_a, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("ARRAY ELEMENTS AFTER Sorting:");
    for (int i = 0; i < n; i++) {
        printf(" %d", a[i]);
    }
    printf("\n");

    printf("Elapsed time: %f ms\n", milliseconds);

    cudaFree(d_a);
    free(a);
    return 0;
}
```
### Experiment 6 - Quick Sort in Parallel
```cpp
#include <stdio.h>
#include <cuda.h>

__device__ void swap(int* a, int i, int j) {
    int temp = a[i];
    a[i] = a[j];
    a[j] = temp;
}

__device__ int partition(int* a, int left, int right) {
    int pivot = a[right];
    int i = left - 1;
    for (int j = left; j < right; j++) {
        if (a[j] < pivot) {
            i++;
            swap(a, i, j);
        }
    }
    swap(a, i + 1, right);
    return i + 1;
}

__global__ void quicksort(int* a, int* stack, int n) {
    int left, right, top, pivot;

    if (threadIdx.x == 0) {
        stack[0] = 0;
        stack[1] = n - 1;
    }
    __syncthreads();

    top = 1;

    while (top >= 0) {
        if (threadIdx.x == 0) {
            right = stack[top--];
            left = stack[top--];
        }
        __syncthreads();

        if (left < right) {
            if (threadIdx.x == 0) {
                pivot = partition(a, left, right);
                stack[++top] = left;
                stack[++top] = pivot - 1;
                stack[++top] = pivot + 1;
                stack[++top] = right;
            }
        }
        __syncthreads();
    }
}

int main() {
    int n;
    printf("Enter the size of the array: ");
    scanf("%d", &n);

    int *h_a = (int*)malloc(n * sizeof(int));
    printf("Enter elements of the array:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_a[i]);
    }

    int *d_a, *d_stack;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_stack, 2 * n * sizeof(int));

    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1);
    dim3 blocksPerGrid(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    quicksort<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_stack, n);
    cudaEventRecord(stop);

    cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\nElapsed time: %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_stack);
    free(h_a);

    return 0;
}
