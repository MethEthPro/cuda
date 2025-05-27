#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

#define HEIGHT 4
#define WIDTH 4

__global__ void RGBtoGrayScale(unsigned char* rgb, unsigned char* gray, int height, int width){
    int x = blockDim.x * blockIdx.x + threadIdx.x; // column
    int y = blockDim.y * blockIdx.y + threadIdx.y; // row

    // Boundary check to make sure we don't access pixels outside the image.
    if (x < width && y < height) {
        int rgb_idx = (y * width + x) * 3;

        unsigned char r = rgb[rgb_idx];
        unsigned char g = rgb[rgb_idx + 1];
        unsigned char b = rgb[rgb_idx + 2];

        gray[y * width + x] = 0.299f * float(r) + 0.587f * float(g) + 0.114f * float(b);

    }
}   


// refer this link https://youtu.be/C_zFhWdM4ic?si=nLzxQu5o-k3esM6i 

__global__ void BoxMeanBlur(unsigned char* input, unsigned char* output, int height, int width, int blur_radius){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sum = 0;
        int count = 0;

        for (int dy = -blur_radius; dy <= blur_radius; dy++) {
            for (int dx = -blur_radius; dx <= blur_radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    sum += input[ny * width + nx];
                    count++;
                }
            }
        }
        if (count > 0) {
            output[y * width + x] = static_cast<unsigned char>(sum / count);
        }
    }
}

__global__ void Convolve2D_custom_kernel(unsigned char* input, unsigned char* output, int width, int height, const float* __restrict__ kernel, 
        int kernel_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_k = kernel_size / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        float weight_kernel_sum = 0.0f;

        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int nx = x + kx;
                int ny = y + ky;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float pixel = static_cast<float>(input[ny * width + nx]);
                    float k_val = kernel[(ky + half_k) * kernel_size + (kx + half_k)];
                    sum += pixel * k_val;
                    weight_kernel_sum += k_val;
                }
            }
        }
        if (weight_kernel_sum > 0.0f) {
            sum /= weight_kernel_sum;
        }
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[y * width + x] = static_cast<unsigned char>(sum);
    }
}

int main(){
    const int RGB_SIZE = HEIGHT * WIDTH * 3;
    const int GRAY_SIZE = HEIGHT * WIDTH;

    unsigned char h_rgb[RGB_SIZE] = {
        255, 0, 0,    0, 255, 0,    0, 0, 255,    255, 255, 0,
        255, 0, 255,  0, 255, 255,  255, 255, 255, 0, 0, 0,
        100, 100, 100, 200, 50, 50,  50, 200, 50,  50, 50, 200,
        10, 20, 30,   40, 50, 60,   70, 80, 90,   100, 110, 120
    };

    unsigned char h_gray[GRAY_SIZE];
    unsigned char h_blur[GRAY_SIZE];
    unsigned char h_conv[GRAY_SIZE];

    unsigned char *d_rgb, *d_gray, *d_blur, *d_conv;
    float *d_kernel;

    cudaMalloc(&d_rgb, RGB_SIZE);
    cudaMalloc(&d_gray, GRAY_SIZE);
    cudaMalloc(&d_blur, GRAY_SIZE);
    cudaMalloc(&d_conv, GRAY_SIZE);

    float h_kernel[] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    const int KERNEL_SIZE = 3;
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rgb, h_rgb, RGB_SIZE, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    RGBtoGrayScale<<<grid, block>>>(d_rgb, d_gray, HEIGHT, WIDTH);
    cudaMemcpy(h_gray, d_gray, GRAY_SIZE, cudaMemcpyDeviceToHost);

    int blur_radius = 1;
    BoxMeanBlur<<<grid, block>>>(d_gray, d_blur, HEIGHT, WIDTH, blur_radius);
    cudaMemcpy(h_blur, d_blur, GRAY_SIZE, cudaMemcpyDeviceToHost);

    Convolve2D_custom_kernel<<<grid, block>>>(d_blur, d_conv, WIDTH, HEIGHT, d_kernel, KERNEL_SIZE);
    cudaMemcpy(h_conv, d_conv, GRAY_SIZE, cudaMemcpyDeviceToHost);

    std::cout << "Grayscale Image:\n";
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << (int)h_gray[y * WIDTH + x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Blurred Image:\n";
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << (int)h_blur[y * WIDTH + x] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Convolved Image:\n";
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            std::cout << (int)h_conv[y * WIDTH + x] << " ";
        }
        std::cout << std::endl;
    }

    // Free only device memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_conv);
    cudaFree(d_kernel);

    return 0;
}
