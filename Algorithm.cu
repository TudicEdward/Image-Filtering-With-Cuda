#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Filtering_Functions.h"

__global__ void Negative_image(int h ,int w, unsigned char* Image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	

	if (x <= w && y <= h)
	{
		int tid = ((y * w) + x)*3;
		Image[tid] = 255 - Image[tid];
		Image[tid+1] = 255 - Image[tid+1];
		Image[tid+2] = 255 - Image[tid+2];
	}
	
}
__global__ void Grayscale_image(int h, int w, unsigned char* Image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x <= w && y <= h)
	{
		int tid = ((y * w) + x) * 3;
		Image[tid] = 0.299 * Image[tid] + 0.587 * Image[tid+1] + 0.114 * Image[tid+2];
		Image[tid + 1] = Image[tid];
		Image[tid + 2] = Image[tid];
	}
	
}
__global__ void Sepia_image(int h, int w, unsigned char* Image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x <= w && y <= h)
	{
		int tid = ((y * w) + x) * 3;
		int Red = 0.393 * Image[tid + 2] + 0.769 * Image[tid + 1] + 0.189 * Image[tid];
		int Green = 0.349 * Image[tid + 2] + 0.686 * Image[tid + 1] + 0.168 * Image[tid];
		int Blue = 0.272 * Image[tid + 2] + 0.534 * Image[tid + 1] + 0.131 * Image[tid];
		Image[tid] = Blue;
		Image[tid + 1] = Green;
		Image[tid + 2] = Red;
	}
	
}
__global__ void Red_image(int h, int w, unsigned char* Image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x <= w && y <= h)
	{
		int tid = ((y * w) + x) * 3;
		Image[tid] = 0;
		Image[tid + 1] = 0;

	}
	
}
__global__ void Green_image(int h, int w, unsigned char* Image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x <= w && y <= h)
	{
		int tid = ((y * w) + x) * 3;
		Image[tid] = 0;
		Image[tid + 2] = 0;
	}
	
}
__global__ void Blue_image(int h, int w, unsigned char* Image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x <= w && y <= h)
	{
		int tid = ((y * w) + x) * 3;
		Image[tid + 1] = 0;
		Image[tid + 2] = 0;
	}
	
}


void Image_Negative(unsigned char* Image, int Height, int Width) {
	unsigned char* Uploaded_Image = NULL;
	dim3 blocks((Width / 16) + 1, (Height / 16) + 1);
	dim3 threads(16, 16);

	cudaMalloc((void**)&Uploaded_Image, Height * Width * 3);

	cudaMemcpy(Uploaded_Image, Image, Height * Width * 3, cudaMemcpyHostToDevice);
	
	Negative_image <<<blocks, threads >>> (Height, Width, Uploaded_Image);

	cudaMemcpy(Image, Uploaded_Image, Height * Width * 3, cudaMemcpyDeviceToHost);

	cudaFree(Uploaded_Image);
}

void Image_Grayscale(unsigned char* Image, int Height, int Width) {
	unsigned char* Uploaded_Image = NULL;
	dim3 blocks((Width / 16)+1, (Height / 16)+1);
	dim3 threads(16, 16);

	cudaMalloc((void**)&Uploaded_Image, Height * Width * 3);

	cudaMemcpy(Uploaded_Image, Image, Height * Width * 3, cudaMemcpyHostToDevice);
	
	Grayscale_image << <blocks, threads >> > (Height, Width, Uploaded_Image);

	cudaMemcpy(Image, Uploaded_Image, Height * Width * 3, cudaMemcpyDeviceToHost);

	cudaFree(Uploaded_Image);
}

void Image_Sepia(unsigned char* Image, int Height, int Width) {
	unsigned char* Uploaded_Image = NULL;
	dim3 blocks((Width / 16) + 1, (Height / 16) + 1);
	dim3 threads(16, 16);

	cudaMalloc((void**)&Uploaded_Image, Height * Width * 3);

	cudaMemcpy(Uploaded_Image, Image, Height * Width * 3, cudaMemcpyHostToDevice);

	Sepia_image << <blocks, threads >> > (Height, Width, Uploaded_Image);

	cudaMemcpy(Image, Uploaded_Image, Height * Width * 3, cudaMemcpyDeviceToHost);

	cudaFree(Uploaded_Image);
}

void Image_Red(unsigned char* Image, int Height, int Width) {
	unsigned char* Uploaded_Image = NULL;
	dim3 blocks((Width / 16) + 1, (Height / 16) + 1);
	dim3 threads(16, 16);

	cudaMalloc((void**)&Uploaded_Image, Height * Width * 3);

	cudaMemcpy(Uploaded_Image, Image, Height * Width * 3, cudaMemcpyHostToDevice);

	Red_image << <blocks, threads >> > (Height, Width, Uploaded_Image);

	cudaMemcpy(Image, Uploaded_Image, Height * Width * 3, cudaMemcpyDeviceToHost);

	cudaFree(Uploaded_Image);
}

void Image_Green(unsigned char* Image, int Height, int Width) {
	unsigned char* Uploaded_Image = NULL;
	dim3 blocks((Width / 16) + 1, (Height / 16) + 1);
	dim3 threads(16, 16);

	cudaMalloc((void**)&Uploaded_Image, Height * Width * 3);

	cudaMemcpy(Uploaded_Image, Image, Height * Width * 3, cudaMemcpyHostToDevice);

	Green_image << <blocks, threads >> > (Height, Width, Uploaded_Image);

	cudaMemcpy(Image, Uploaded_Image, Height * Width * 3, cudaMemcpyDeviceToHost);

	cudaFree(Uploaded_Image);
}

void Image_Blue(unsigned char* Image, int Height, int Width) {
	unsigned char* Uploaded_Image = NULL;
	dim3 blocks((Width / 16) + 1, (Height / 16) + 1);
	dim3 threads(16, 16);

	cudaMalloc((void**)&Uploaded_Image, Height * Width * 3);

	cudaMemcpy(Uploaded_Image, Image, Height * Width * 3, cudaMemcpyHostToDevice);

	Blue_image << <blocks, threads >> > (Height, Width, Uploaded_Image);

	cudaMemcpy(Image, Uploaded_Image, Height * Width * 3, cudaMemcpyDeviceToHost);

	cudaFree(Uploaded_Image);
}