// Create other necessary functions here
#include <cuda_runtime.h>

__global__ void convolutionKernel(int input_row, int input_col, int *input,
                                 int kernel_row, int kernel_col, int *kernel,
                                 int output_row, int output_col,
                                 long long unsigned int *output) {
    int output_i = blockIdx.y * blockDim.y + threadIdx.y;
    int output_j = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (output_i < output_row && output_j < output_col) {
        for (int kernel_i = 0; kernel_i < kernel_row; kernel_i++) {
            for (int kernel_j = 0; kernel_j < kernel_col; kernel_j++) {
                int input_i = (output_i + 2 * kernel_i) % input_row;
                int input_j = (output_j + 2 * kernel_j) % input_col;
                output[output_i * output_col + output_j] +=
                    input[input_i * input_col + input_j] *
                    kernel[kernel_i * kernel_col + kernel_j];
            }
        }
    }
}
 
void gpuThread(int input_row, int input_col, int *input,
                     int kernel_row, int kernel_col, int *kernel,
                     int output_row, int output_col,
                     long long unsigned int *output) {
    int *d_input, *d_kernel;
    long long unsigned int *d_output;
 
    // Allocate memory on GPU
    cudaMalloc((void **)&d_input, sizeof(int) * input_row * input_col);
    cudaMalloc((void **)&d_kernel, sizeof(int) * kernel_row * kernel_col);
    cudaMalloc((void **)&d_output, sizeof(long long unsigned int) * output_row * output_col);
 
    // Copy input and kernel to GPU
    cudaMemcpy(d_input, input, sizeof(int) * input_row * input_col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(int) * kernel_row * kernel_col, cudaMemcpyHostToDevice);
 
    // Define grid and block sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((output_col + blockSize.x - 1) / blockSize.x, (output_row + blockSize.y - 1) / blockSize.y);
 
    // Launch CUDA kernel
    convolutionKernel<<<gridSize, blockSize>>>(input_row, input_col, d_input,
                                               kernel_row, kernel_col, d_kernel,
                                               output_row, output_col,
                                               d_output);
 
    // Copy output from GPU to CPU
    cudaMemcpy(output, d_output, sizeof(long long unsigned int) * output_row * output_col, cudaMemcpyDeviceToHost);
 
    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}