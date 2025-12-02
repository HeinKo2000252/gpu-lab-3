#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>

using namespace std;

#define BLOCK_DIM 16

struct Image {
    int width;
    int height;
    png_bytep data;
};

enum FilterType {
    FILTER_BLUR,
    FILTER_EDGE,
    FILTER_DENOISE
};

__constant__ float d_kernel[9];

/// 1. PNG ဖတ်တဲ့ Function
Image readPNG(const char* filename) {
    png_image image;
    memset(&image, 0, sizeof(image));        // struct ကို zero လုပ်တယ်
    image.version = PNG_IMAGE_VERSION;        // libpng version သတ်မှတ်တယ်

    if (!png_image_begin_read_from_file(&image, filename)) {
        fprintf(stderr, "Error reading PNG: %s\n", image.message);
        exit(1);                            // Error ဆိုရင် program ရပ်
    }

    image.format = PNG_FORMAT_RGBA;                // RGBA format ပြောင်းမယ်
    png_bytep buffer = (png_bytep)malloc(PNG_IMAGE_SIZE(image));     // Memory allocate
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed for image data\n");
        png_image_free(&image);
        exit(1);
    }

     // ပုံ data တွေကို buffer ထဲဖတ်မယ်
    if (!png_image_finish_read(&image, nullptr, buffer, 0, nullptr)) {
        fprintf(stderr, "Error reading PNG data: %s\n", image.message);
        free(buffer);
        png_image_free(&image);
        exit(1);
    }

    // ရလာတဲ့ data တွေကို Image struct ထဲထည့်မယ်
    Image result;
    result.width = image.width;
    result.height = image.height;
    result.data = buffer;

    png_image_free(&image);        // Temporary memory ရှင်းမယ်
    return result;
}

/// 2. PNG ရေးတဲ့ Function
void writePNG(const char* filename, const Image& img) {
    png_image image;
    memset(&image, 0, sizeof(image));        // Zero initialize
    image.version = PNG_IMAGE_VERSION;
    image.width = img.width;
    image.height = img.height;
    image.format = PNG_FORMAT_RGBA;        // RGBA format နဲ့ရေးမယ်

    printf("Writing PNG file: %s\n", filename);

    // File ကို write လုပ်မယ်
    if (!png_image_write_to_file(&image, filename, 0, img.data, 0, nullptr)) {
        fprintf(stderr, "Error writing PNG: %s\n", image.message);
        png_image_free(&image);
        exit(1);
    }

    png_image_free(&image);
    printf("PNG written successfully\n");
}

/// 3. Memory ရှင်းတဲ့ Function
void freeImage(Image& img) {
    if (img.data) {
        free(img.data);        // Image data ကို free လုပ်မယ်
        img.data = nullptr;    // Pointer ကို null လုပ်မယ်
    }
    img.width = img.height = 0;        // Dimensions ကို zero လုပ်မယ်
}

/// 4. CUDA Kernel Function - 3x3 Convolution Filter
__global__ void kernel_filter(const unsigned char* input, unsigned char* output, 
                              int width, int height, int channels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;        // Boundary check

    int idx = (y * width + x) * channels;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;

    // 3x3 Convolution
    for (int ky = -1; ky <= 1; ++ky) {
        int sy = (int)y + ky;
        if (sy < 0) sy = 0;
        if (sy >= height) sy = height - 1;

        for (int kx = -1; kx <= 1; ++kx) {
            int sx = (int)x + kx;
            if (sx < 0) sx = 0;
            if (sx >= width) sx = width - 1;

            int sidx = (sy * width + sx) * channels;
            float k = d_kernel[(ky + 1) * 3 + (kx + 1)];

            accumR += k * input[sidx + 0];
            accumG += k * input[sidx + 1];
            accumB += k * input[sidx + 2];
        }
    }

    int r = (int)roundf(accumR);
    int g = (int)roundf(accumG);
    int b = (int)roundf(accumB);

    if (r < 0) r = 0; if (r > 255) r = 255;
    if (g < 0) g = 0; if (g > 255) g = 255;
    if (b < 0) b = 0; if (b > 255) b = 255;

    output[idx + 0] = (unsigned char)r;
    output[idx + 1] = (unsigned char)g;
    output[idx + 2] = (unsigned char)b;

    if (channels > 3) {
        output[idx + 3] = input[idx + 3];
    }
}

/// 5. Filter Type Parser
FilterType parse_filter(const string& name) {
    if (name == "blur") return FILTER_BLUR;
    if (name == "edge") return FILTER_EDGE;
    if (name == "denoise") return FILTER_DENOISE;

    cerr << "Unknown filter: " << name << endl;
    cerr << "Use one of: blur | edge | denoise" << endl;
    exit(1);
}

/// 6. Kernel Matrix Generator
void get_kernel(FilterType type, float kernel[9]) {
    if (type == FILTER_BLUR) {
        float v = 1.0f / 9.0f;
        for (int i = 0; i < 9; ++i) kernel[i] = v;
    } else if (type == FILTER_EDGE) {
        float tmp[9] = {
             0.0f, -1.0f,  0.0f,
            -1.0f,  4.0f, -1.0f,
             0.0f, -1.0f,  0.0f
        };
        for (int i = 0; i < 9; ++i) kernel[i] = tmp[i];
    } else if (type == FILTER_DENOISE) {
        float tmp[9] = {
            1.f, 2.f, 1.f,
            2.f, 4.f, 2.f,
            1.f, 2.f, 1.f
        };
        for (int i = 0; i < 9; ++i) kernel[i] = tmp[i] / 16.0f;
    }
}

/// 7. Main Function
int main(int argc, char** argv) {
    // Command line arguments check
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " input.png output.png [blur|edge|denoise]" << endl;
        return 1;
    }

    const char* input_name = argv[1];
    const char* output_name = argv[2];
    string filter_name = argv[3];

    cudaError_t cuda_error;
    int device_count = 0;
    cuda_error = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess) {
        cout << "cudaGetDeviceCount failed: " << cudaGetErrorString(cuda_error) << endl;
        return 1;
    }

    if (device_count < 2) {
        cout << "This program requires at least 2 CUDA devices, found: " << device_count << endl;
        return 1;
    }

    // GPU Properties ပြခြင်း
    for (int dev = 0; dev < 2; ++dev) {
        cudaDeviceProp prop{};
        cuda_error = cudaGetDeviceProperties(&prop, dev);
        if (cuda_error != cudaSuccess) {
            cout << "Error getting device properties for device " << dev << ": " << cudaGetErrorString(cuda_error) << endl;
            return 1;
        }

        cout << "Device name: " << prop.name << endl;
        cout << "Number of multiprocessors: " << prop.multiProcessorCount << endl;
        cout << "Global memory size: " << prop.totalGlobalMem << " bytes" << endl;
        cout << "Max threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "Max grid size: " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << endl;
        cout << "Max block dimensions: " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << endl;
        cout << endl;
    }

    // Image Data တွက်ချက်ခြင်း
    Image input_img = readPNG(input_name);
    int width = input_img.width;
    int height = input_img.height;
    int channels = 4;

    cout << "Loaded image: " << input_name << " (" << width << " x " << height << "), channels = " << channels << endl;

    FilterType filter_type = parse_filter(filter_name);
    float h_kernel[9];
    get_kernel(filter_type, h_kernel);

    size_t row_bytes = (size_t)width * channels * sizeof(unsigned char);
    int half_h_0 = height / 2;
    int half_h_1 = height - half_h_0;

    size_t bytes_out_0 = (size_t)half_h_0 * row_bytes;
    size_t bytes_out_1 = (size_t)half_h_1 * row_bytes;

    int top_local_h = (half_h_0 > 0 && height > 1) ? (half_h_0 + 1) : half_h_0;
    int bottom_local_h = (half_h_1 > 0 && height > 1) ? (half_h_1 + 1) : half_h_1;

    size_t bytes_in_0 = (size_t)top_local_h * row_bytes;
    size_t bytes_in_1 = (size_t)bottom_local_h * row_bytes;

    unsigned char* host_input = input_img.data;
    unsigned char* host_output = (unsigned char*)malloc((size_t)height * row_bytes);

    if (!host_output) {
        cerr << "Failed to allocate host_output" << endl;
        freeImage(input_img);
        return 1;
    }

    unsigned char* host_top = host_input;
    unsigned char* host_bottom = host_input + (size_t)(half_h_0 - 1) * row_bytes;
    
    // ===== CUDA Events and Streams ပြင်ထား =====
    cudaEvent_t evH2DStart_0, evH2DStop_0;
    cudaEvent_t evH2DStart_1, evH2DStop_1;
    cudaEvent_t evKernelStart_0, evKernelStop_0;
    cudaEvent_t evKernelStart_1, evKernelStop_1;
    cudaEvent_t evD2HStart_0, evD2HStop_0;
    cudaEvent_t evD2HStart_1, evD2HStop_1;
    cudaEvent_t evTotalStart, evTotalStop;
    
    // ပြင်ထား: CUDA Streams for parallel execution
    cudaStream_t stream0, stream1;

    int dev_0 = 0;
    int dev_1 = 1;

    // GPU 0 setup with stream
    cudaSetDevice(dev_0);
    cudaEventCreate(&evH2DStart_0);
    cudaEventCreate(&evH2DStop_0);
    cudaEventCreate(&evKernelStart_0);
    cudaEventCreate(&evKernelStop_0);
    cudaEventCreate(&evD2HStart_0);
    cudaEventCreate(&evD2HStop_0);
    cudaStreamCreate(&stream0);  // ပြင်ထား: stream create for GPU 0

    // GPU 1 setup with stream
    cudaSetDevice(dev_1);
    cudaEventCreate(&evH2DStart_1);
    cudaEventCreate(&evH2DStop_1);
    cudaEventCreate(&evKernelStart_1);
    cudaEventCreate(&evKernelStop_1);
    cudaEventCreate(&evD2HStart_1);
    cudaEventCreate(&evD2HStop_1);
    cudaStreamCreate(&stream1);  // ပြင်ထား: stream create for GPU 1

    // Total events
    cudaEventCreate(&evTotalStart);
    cudaEventCreate(&evTotalStop);

    cudaEventRecord(evTotalStart, 0);

    // ===== GPU 0: Asynchronous Operations with stream0 ပြင်ထား =====
    unsigned char* dev_in_0 = nullptr;
    unsigned char* dev_out_0 = nullptr;

    if (top_local_h > 0) {
        cudaSetDevice(dev_0);
        
        cuda_error = cudaMalloc((void**)&dev_in_0, bytes_in_0);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMalloc(dev_in_0) failed: " << cudaGetErrorString(cuda_error) << endl;
            freeImage(input_img);
            free(host_output);
            return 1;
        }

        cuda_error = cudaMalloc((void**)&dev_out_0, bytes_in_0);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMalloc(dev_out_0) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            freeImage(input_img);
            free(host_output);
            return 1;
        }

        cuda_error = cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMemcpyToSymbol (dev_0) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            cudaFree(dev_out_0);
            freeImage(input_img);
            free(host_output);
            return 1;
        }

        // ပြင်ထား: Asynchronous H2D copy with stream0
        cudaEventRecord(evH2DStart_0, stream0);
        cuda_error = cudaMemcpyAsync(dev_in_0, host_top, bytes_in_0, 
                                     cudaMemcpyHostToDevice, stream0);
        cudaEventRecord(evH2DStop_0, stream0);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMemcpyAsync host -> device (dev_0) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            cudaFree(dev_out_0);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
    }

    // ===== GPU 1: Asynchronous Operations with stream1 ပြင်ထား =====
    unsigned char* dev_in_1 = nullptr;
    unsigned char* dev_out_1 = nullptr;

    if (bottom_local_h > 0) {
        cudaSetDevice(dev_1);

        cuda_error = cudaMalloc((void**)&dev_in_1, bytes_in_1);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMalloc(dev_in_1) failed: " << cudaGetErrorString(cuda_error) << endl;
            freeImage(input_img);
            free(host_output);
            return 1;
        }

        cuda_error = cudaMalloc((void**)&dev_out_1, bytes_in_1);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMalloc(dev_out_1) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }

        cuda_error = cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float), 0, cudaMemcpyHostToDevice);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMemcpyToSymbol (dev_1) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_1);
            cudaFree(dev_out_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
        
        // ပြင်ထား: Asynchronous H2D copy with stream1
        cudaEventRecord(evH2DStart_1, stream1);
        cuda_error = cudaMemcpyAsync(dev_in_1, host_bottom, bytes_in_1, 
                                     cudaMemcpyHostToDevice, stream1);
        cudaEventRecord(evH2DStop_1, stream1);
        if (cuda_error != cudaSuccess) {
            cout << "cudaMemcpyAsync host -> device (dev_1) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_1);
            cudaFree(dev_out_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
    }

    // ===== PARALLEL KERNEL EXECUTION ပြင်ထား =====
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);

    // GPU 0 kernel with stream0 ပြင်ထား
    if (top_local_h > 0) {
        cudaSetDevice(dev_0);

        dim3 gridSize_0(
            (width + blockSize.x - 1) / blockSize.x,
            (top_local_h + blockSize.y - 1) / blockSize.y
        );

        // ပြင်ထား: Asynchronous kernel launch with stream0
        cudaEventRecord(evKernelStart_0, stream0);
        kernel_filter<<<gridSize_0, blockSize, 0, stream0>>>(dev_in_0, dev_out_0, 
                                                           width, top_local_h, channels);
        cudaEventRecord(evKernelStop_0, stream0);

        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cout << "Kernel launch failed on dev_0: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            cudaFree(dev_out_0);
            cudaFree(dev_in_1);
            cudaFree(dev_out_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
    }

    // GPU 1 kernel with stream1 ပြင်ထား (PARALLEL with GPU 0)
    if (bottom_local_h > 0) {
        cudaSetDevice(dev_1);

        dim3 gridSize_1(
            (width + blockSize.x - 1) / blockSize.x,
            (bottom_local_h + blockSize.y - 1) / blockSize.y
        );

        // ပြင်ထား: Asynchronous kernel launch with stream1
        cudaEventRecord(evKernelStart_1, stream1);
        kernel_filter<<<gridSize_1, blockSize, 0, stream1>>>(dev_in_1, dev_out_1, 
                                                           width, bottom_local_h, channels);
        cudaEventRecord(evKernelStop_1, stream1);

        cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            cout << "Kernel launch failed on dev_1: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            cudaFree(dev_out_0);
            cudaFree(dev_in_1);
            cudaFree(dev_out_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
    }

    // ပြင်ထား: Remove cudaDeviceSynchronize() for parallel execution
    
    // ===== PARALLEL DATA COPY BACK ပြင်ထား =====
    
    // GPU 0: Asynchronous D2H with stream0 ပြင်ထား
    if (top_local_h > 0) {
        cudaSetDevice(dev_0);
        
        // ပြင်ထား: Asynchronous D2H copy with stream0
        cudaEventRecord(evD2HStart_0, stream0);
        cuda_error = cudaMemcpyAsync(host_output, dev_out_0, bytes_out_0, 
                                     cudaMemcpyDeviceToHost, stream0);
        cudaEventRecord(evD2HStop_0, stream0);

        if (cuda_error != cudaSuccess) {
            cout << "cudaMemcpyAsync device->host (dev_0) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            cudaFree(dev_out_0);
            cudaFree(dev_in_1);
            cudaFree(dev_out_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
    }

    // GPU 1: Asynchronous D2H with stream1 ပြင်ထား (PARALLEL with GPU 0)
    if (bottom_local_h > 0) {
        cudaSetDevice(dev_1);

        const unsigned char* dev_out_1_inner = dev_out_1 + row_bytes; 

        // ပြင်ထား: Asynchronous D2H copy with stream1
        cudaEventRecord(evD2HStart_1, stream1);
        cuda_error = cudaMemcpyAsync(host_output + bytes_out_0, dev_out_1_inner, bytes_out_1, 
                                     cudaMemcpyDeviceToHost, stream1);
        cudaEventRecord(evD2HStop_1, stream1);

        if (cuda_error != cudaSuccess) {
            cout << "cudaMemcpyAsync device->host (dev_1) failed: " << cudaGetErrorString(cuda_error) << endl;
            cudaFree(dev_in_0);
            cudaFree(dev_out_0);
            cudaFree(dev_in_1);
            cudaFree(dev_out_1);
            freeImage(input_img);
            free(host_output);
            return 1;
        }
    }

    // ===== SYNCHRONIZE STREAMS (NOT DEVICES) ပြင်ထား =====
    // Wait for both streams to complete (GPU 0 and GPU 1 run in parallel)
    if (top_local_h > 0) {
        cudaSetDevice(dev_0);
        cudaStreamSynchronize(stream0);  // ပြင်ထား: Only wait for stream0
    }
    if (bottom_local_h > 0) {
        cudaSetDevice(dev_1);
        cudaStreamSynchronize(stream1);  // ပြင်ထား: Only wait for stream1
    }

    cudaEventRecord(evTotalStop, 0);
    cudaEventSynchronize(evTotalStop);

    // ===== TIMING CALCULATIONS =====
    float timeH2D_0 = 0.0f;
    float timeH2D_1 = 0.0f;
    float timeKernel_0 = 0.0f;
    float timeKernel_1 = 0.0f;
    float timeD2H_0 = 0.0f;
    float timeD2H_1 = 0.0f;
    float timeTotal = 0.0f;

    if (half_h_0 > 0) {
        cudaEventElapsedTime(&timeH2D_0, evH2DStart_0, evH2DStop_0);
        cudaEventElapsedTime(&timeKernel_0, evKernelStart_0, evKernelStop_0);
        cudaEventElapsedTime(&timeD2H_0, evD2HStart_0, evD2HStop_0);
    }
    if (half_h_1 > 0) {
        cudaEventElapsedTime(&timeH2D_1, evH2DStart_1, evH2DStop_1);
        cudaEventElapsedTime(&timeKernel_1, evKernelStart_1, evKernelStop_1);
        cudaEventElapsedTime(&timeD2H_1, evD2HStart_1, evD2HStop_1);
    }
    cudaEventElapsedTime(&timeTotal, evTotalStart, evTotalStop);

    cout << fixed << setprecision(3);
    cout << "GPU 0:" << endl;
    cout << "Host -> Device copy time: " << timeH2D_0 << " ms" << endl;
    cout << "Kernel execution time: " << timeKernel_0 << " ms" << endl;
    cout << "Device -> Host copy time: " << timeD2H_0 << " ms" << endl;
    cout << "GPU 1:" << endl;
    cout << "Host -> Device copy time: " << timeH2D_1 << " ms" << endl;
    cout << "Kernel execution time: " << timeKernel_1 << " ms" << endl;
    cout << "Device -> Host copy time: " << timeD2H_1 << " ms" << endl;
    cout << "Total GPU time (parallel): " << timeTotal << " ms" << endl;

    // ===== CLEANUP =====
    // 1. CUDA events တွေ destroy လုပ်
    cudaEventDestroy(evH2DStart_0);
    cudaEventDestroy(evH2DStop_0);
    cudaEventDestroy(evH2DStart_1);
    cudaEventDestroy(evH2DStop_1);
    cudaEventDestroy(evKernelStart_0);
    cudaEventDestroy(evKernelStop_0);
    cudaEventDestroy(evKernelStart_1);
    cudaEventDestroy(evKernelStop_1);
    cudaEventDestroy(evD2HStart_0);
    cudaEventDestroy(evD2HStop_0);
    cudaEventDestroy(evD2HStart_1);
    cudaEventDestroy(evD2HStop_1);
    cudaEventDestroy(evTotalStart);
    cudaEventDestroy(evTotalStop);

    // ပြင်ထား: Destroy streams
    if (top_local_h > 0) {
        cudaSetDevice(dev_0);
        cudaStreamDestroy(stream0);
    }
    if (bottom_local_h > 0) {
        cudaSetDevice(dev_1);
        cudaStreamDestroy(stream1);
    }

    Image output_img;
    output_img.width = width;
    output_img.height = height;
    output_img.data = host_output;

    // 2. PNG file ရေး
    writePNG(output_name, output_img);

    // 3. GPU memory free လုပ်
    if (half_h_0 > 0) {
        cudaSetDevice(dev_0);
        cudaFree(dev_in_0);
        cudaFree(dev_out_0);
    }
    if (half_h_1 > 0) {
        cudaSetDevice(dev_1);
        cudaFree(dev_in_1);
        cudaFree(dev_out_1);
    }

    // 4. Host memory free လုပ်
    freeImage(input_img);
    freeImage(output_img);

    return 0;
}
