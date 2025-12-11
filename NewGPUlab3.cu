#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

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

enum ExecutionMode {
    MODE_SINGLE,
    MODE_DUAL,
    MODE_COMPARE
};

__constant__ float d_kernel[9];

// ================== PNG FUNCTIONS ==================
Image readPNG(const char* filename) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, filename)) {
        fprintf(stderr, "Error reading PNG: %s\n", image.message);
        exit(1);
    }

    image.format = PNG_FORMAT_RGBA;
    png_bytep buffer = (png_bytep)malloc(PNG_IMAGE_SIZE(image));
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed for image data\n");
        png_image_free(&image);
        exit(1);
    }

    if (!png_image_finish_read(&image, nullptr, buffer, 0, nullptr)) {
        fprintf(stderr, "Error reading PNG data: %s\n", image.message);
        free(buffer);
        png_image_free(&image);
        exit(1);
    }

    Image result;
    result.width = image.width;
    result.height = image.height;
    result.data = buffer;

    png_image_free(&image);
    return result;
}

void writePNG(const char* filename, const Image& img) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    image.width = img.width;
    image.height = img.height;
    image.format = PNG_FORMAT_RGBA;

    if (!png_image_write_to_file(&image, filename, 0, img.data, 0, nullptr)) {
        fprintf(stderr, "Error writing PNG: %s\n", image.message);
        png_image_free(&image);
        exit(1);
    }

    png_image_free(&image);
}

void freeImage(Image& img) {
    if (img.data) {
        free(img.data);
        img.data = nullptr;
    }
    img.width = img.height = 0;
}

// ================== CUDA KERNEL ==================
__global__ void kernel_filter(const unsigned char* input, unsigned char* output, 
                              int width, int height, int channels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;

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

// ================== HELPER FUNCTIONS ==================
FilterType parse_filter(const string& name) {
    if (name == "blur") return FILTER_BLUR;
    if (name == "edge") return FILTER_EDGE;
    if (name == "denoise") return FILTER_DENOISE;

    cerr << "Unknown filter: " << name << endl;
    cerr << "Use one of: blur | edge | denoise" << endl;
    exit(1);
}

void get_kernel(FilterType type, float kernel[9]) {
    if (type == FILTER_BLUR) {
        float v = 1.0f / 9.0f;
        for (int i = 0; i < 9; ++i) kernel[i] = v;
    } else if (type == FILTER_EDGE) {
        float tmp[9] = {0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f};
        for (int i = 0; i < 9; ++i) kernel[i] = tmp[i];
    } else if (type == FILTER_DENOISE) {
        float tmp[9] = {1.f, 2.f, 1.f, 2.f, 4.f, 2.f, 1.f, 2.f, 1.f};
        for (int i = 0; i < 9; ++i) kernel[i] = tmp[i] / 16.0f;
    }
}

ExecutionMode parse_mode(const string& mode) {
    if (mode == "single") return MODE_SINGLE;
    if (mode == "dual") return MODE_DUAL;
    if (mode == "compare") return MODE_COMPARE;
    
    cerr << "Unknown mode: " << mode << endl;
    cerr << "Use one of: single | dual | compare" << endl;
    exit(1);
}

// ================== SINGLE GPU EXECUTION ==================
float run_single_gpu(const char* input_name, const char* output_name, FilterType filter_type) {
    cout << "\n=== SINGLE GPU EXECUTION ===" << endl;
    
    cudaSetDevice(0);
    
    Image input_img = readPNG(input_name);
    int width = input_img.width;
    int height = input_img.height;
    int channels = 4;
    
    cout << "Image: " << width << "x" << height << ", channels: " << channels << endl;
    cout << "Using GPU 0 only" << endl;
    
    float h_kernel[9];
    get_kernel(filter_type, h_kernel);
    
    size_t total_bytes = (size_t)width * height * channels * sizeof(unsigned char);
    unsigned char* host_output = (unsigned char*)malloc(total_bytes);
    
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in, total_bytes);
    cudaMalloc(&d_out, total_bytes);
    
    cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float));
    
    // Timing events
    cudaEvent_t start_total, stop_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    
    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t kernel_start, kernel_stop;
    cudaEvent_t d2h_start, d2h_stop;
    
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    
    // Host -> Device
    cudaEventRecord(h2d_start);
    cudaMemcpy(d_in, input_img.data, total_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    
    // Kernel execution
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    cudaEventRecord(kernel_start);
    cudaEventRecord(start_total);
    
    kernel_filter<<<gridSize, blockSize>>>(d_in, d_out, width, height, channels);
    
    cudaEventRecord(kernel_stop);
    
    // Device -> Host
    cudaEventRecord(d2h_start);
    cudaMemcpy(host_output, d_out, total_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop);
    
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    
    // Calculate times
    float time_h2d, time_kernel, time_d2h, time_total;
    cudaEventElapsedTime(&time_h2d, h2d_start, h2d_stop);
    cudaEventElapsedTime(&time_kernel, kernel_start, kernel_stop);
    cudaEventElapsedTime(&time_d2h, d2h_start, d2h_stop);
    cudaEventElapsedTime(&time_total, start_total, stop_total);
    
    cout << fixed << setprecision(3);
    cout << "Host -> Device: " << time_h2d << " ms" << endl;
    cout << "Kernel execution: " << time_kernel << " ms" << endl;
    cout << "Device -> Host: " << time_d2h << " ms" << endl;
    cout << "Total time: " << time_total << " ms" << endl;
    
    // Save result
    Image output_img;
    output_img.width = width;
    output_img.height = height;
    output_img.data = host_output;
    
    writePNG(output_name, output_img);
    cout << "Saved: " << output_name << endl;
    
    // Cleanup
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    
    cudaFree(d_in);
    cudaFree(d_out);
    freeImage(input_img);
    free(output_img.data);
    
    return time_total;
}

// ================== DUAL GPU EXECUTION ==================
float run_dual_gpu(const char* input_name, const char* output_name, FilterType filter_type) {
    cout << "\n=== DUAL GPU EXECUTION ===" << endl;
    
    Image input_img = readPNG(input_name);
    int width = input_img.width;
    int height = input_img.height;
    int channels = 4;
    
    cout << "Image: " << width << "x" << height << ", channels: " << channels << endl;
    cout << "Using GPU 0 and GPU 1" << endl;
    
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
    
    unsigned char* host_top = host_input;
    unsigned char* host_bottom = host_input + (size_t)(half_h_0 - 1) * row_bytes;
    
    // Events for GPU 0
    cudaSetDevice(0);
    cudaEvent_t evH2DStart_0, evH2DStop_0;
    cudaEvent_t evKernelStart_0, evKernelStop_0;
    cudaEvent_t evD2HStart_0, evD2HStop_0;
    cudaEventCreate(&evH2DStart_0);
    cudaEventCreate(&evH2DStop_0);
    cudaEventCreate(&evKernelStart_0);
    cudaEventCreate(&evKernelStop_0);
    cudaEventCreate(&evD2HStart_0);
    cudaEventCreate(&evD2HStop_0);
    
    // Events for GPU 1
    cudaSetDevice(1);
    cudaEvent_t evH2DStart_1, evH2DStop_1;
    cudaEvent_t evKernelStart_1, evKernelStop_1;
    cudaEvent_t evD2HStart_1, evD2HStop_1;
    cudaEventCreate(&evH2DStart_1);
    cudaEventCreate(&evH2DStop_1);
    cudaEventCreate(&evKernelStart_1);
    cudaEventCreate(&evKernelStop_1);
    cudaEventCreate(&evD2HStart_1);
    cudaEventCreate(&evD2HStop_1);
    
    // Total time event
    cudaEvent_t evTotalStart, evTotalStop;
    cudaEventCreate(&evTotalStart);
    cudaEventCreate(&evTotalStop);
    
    cudaEventRecord(evTotalStart, 0);
    
    // GPU 0 processing
    unsigned char *dev_in_0 = nullptr, *dev_out_0 = nullptr;
    if (top_local_h > 0) {
        cudaSetDevice(0);
        cudaMalloc(&dev_in_0, bytes_in_0);
        cudaMalloc(&dev_out_0, bytes_in_0);
        cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float));
        
        cudaEventRecord(evH2DStart_0, 0);
        cudaMemcpy(dev_in_0, host_top, bytes_in_0, cudaMemcpyHostToDevice);
        cudaEventRecord(evH2DStop_0, 0);
        cudaEventSynchronize(evH2DStop_0);
        
        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
        dim3 gridSize_0((width + blockSize.x - 1) / blockSize.x,
                       (top_local_h + blockSize.y - 1) / blockSize.y);
        
        cudaEventRecord(evKernelStart_0, 0);
        kernel_filter<<<gridSize_0, blockSize>>>(dev_in_0, dev_out_0, width, top_local_h, channels);
        cudaEventRecord(evKernelStop_0, 0);
        
        cudaEventRecord(evD2HStart_0, 0);
        cudaMemcpy(host_output, dev_out_0, bytes_out_0, cudaMemcpyDeviceToHost);
        cudaEventRecord(evD2HStop_0, 0);
    }
    
    // GPU 1 processing
    unsigned char *dev_in_1 = nullptr, *dev_out_1 = nullptr;
    if (bottom_local_h > 0) {
        cudaSetDevice(1);
        cudaMalloc(&dev_in_1, bytes_in_1);
        cudaMalloc(&dev_out_1, bytes_in_1);
        cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float));
        
        cudaEventRecord(evH2DStart_1, 0);
        cudaMemcpy(dev_in_1, host_bottom, bytes_in_1, cudaMemcpyHostToDevice);
        cudaEventRecord(evH2DStop_1, 0);
        cudaEventSynchronize(evH2DStop_1);
        
        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
        dim3 gridSize_1((width + blockSize.x - 1) / blockSize.x,
                       (bottom_local_h + blockSize.y - 1) / blockSize.y);
        
        cudaEventRecord(evKernelStart_1, 0);
        kernel_filter<<<gridSize_1, blockSize>>>(dev_in_1, dev_out_1, width, bottom_local_h, channels);
        cudaEventRecord(evKernelStop_1, 0);
        
        const unsigned char* dev_out_1_inner = dev_out_1 + row_bytes;
        cudaEventRecord(evD2HStart_1, 0);
        cudaMemcpy(host_output + bytes_out_0, dev_out_1_inner, bytes_out_1, cudaMemcpyDeviceToHost);
        cudaEventRecord(evD2HStop_1, 0);
    }
    
    // Synchronize both GPUs
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    
    cudaEventRecord(evTotalStop, 0);
    cudaEventSynchronize(evTotalStop);
    
    // Calculate times
    float timeH2D_0 = 0.0f, timeKernel_0 = 0.0f, timeD2H_0 = 0.0f;
    float timeH2D_1 = 0.0f, timeKernel_1 = 0.0f, timeD2H_1 = 0.0f;
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
    
    // Print results
    cout << fixed << setprecision(3);
    cout << "\nGPU 0 results:" << endl;
    cout << "  Host->Device: " << timeH2D_0 << " ms" << endl;
    cout << "  Kernel: " << timeKernel_0 << " ms" << endl;
    cout << "  Device->Host: " << timeD2H_0 << " ms" << endl;
    
    cout << "\nGPU 1 results:" << endl;
    cout << "  Host->Device: " << timeH2D_1 << " ms" << endl;
    cout << "  Kernel: " << timeKernel_1 << " ms" << endl;
    cout << "  Device->Host: " << timeD2H_1 << " ms" << endl;
    
    cout << "\nTotal dual GPU time: " << timeTotal << " ms" << endl;
    
    // Determine bottleneck (slowest GPU)
    float bottleneck_time = max(timeKernel_0, timeKernel_1);
    cout << "Bottleneck (slowest GPU kernel time): " << bottleneck_time << " ms" << endl;
    
    // Save result
    Image output_img;
    output_img.width = width;
    output_img.height = height;
    output_img.data = host_output;
    
    writePNG(output_name, output_img);
    cout << "Saved: " << output_name << endl;
    
    // Cleanup
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
    
    if (dev_in_0) cudaFree(dev_in_0);
    if (dev_out_0) cudaFree(dev_out_0);
    if (dev_in_1) cudaFree(dev_in_1);
    if (dev_out_1) cudaFree(dev_out_1);
    
    freeImage(input_img);
    free(host_output);
    
    return timeTotal;
}

// ================== COMPARISON FUNCTION ==================
void run_comparison(const char* input_name, FilterType filter_type) {
    cout << "\n=== COMPARISON MODE ===" << endl;
    cout << "This will run both single and dual GPU modes and compare results" << endl;
    
    // Run single GPU
    float single_time = run_single_gpu(input_name, "output_single.png", filter_type);
    
    // Run dual GPU
    float dual_time = run_dual_gpu(input_name, "output_dual.png", filter_type);
    
    // Compare results
    cout << "\n=== COMPARISON RESULTS ===" << endl;
    cout << fixed << setprecision(3);
    cout << "Single GPU time: " << single_time << " ms" << endl;
    cout << "Dual GPU time: " << dual_time << " ms" << endl;
    
    if (dual_time > 0) {
        float speedup = single_time / dual_time;
        float efficiency = (speedup / 2.0f) * 100.0f;
        
        cout << "\nSpeedup: " << speedup << "x" << endl;
        cout << "Efficiency: " << efficiency << "%" << endl;
        
        cout << "\n=== ANALYSIS ===" << endl;
        if (speedup > 1.0f) {
            cout << "✓ Dual GPU is " << speedup << " times faster than single GPU" << endl;
            if (efficiency > 70.0f) {
                cout << "✓ Good efficiency (" << efficiency << "%)" << endl;
            } else if (efficiency > 50.0f) {
                cout << "○ Moderate efficiency (" << efficiency << "%)" << endl;
                cout << "  - Possible load imbalance between GPUs" << endl;
            } else {
                cout << "✗ Low efficiency (" << efficiency << "%)" << endl;
                cout << "  - Possible synchronization overhead" << endl;
                cout << "  - Check for Python processes on GPU1 (8230MiB usage)" << endl;
            }
        } else {
            cout << "✗ Dual GPU is slower than single GPU" << endl;
            cout << "  - Possible reasons:" << endl;
            cout << "    1. Small image size (not enough work for parallelization)" << endl;
            cout << "    2. High overhead for data splitting" << endl;
            cout << "    3. GPU1 has Python process (8230MiB) slowing it down" << endl;
        }
    }
}

// ================== MAIN FUNCTION ==================
int main(int argc, char** argv) {
    cout << "=== CUDA Image Filter with Single/Dual GPU Comparison ===" << endl;
    cout << "=== замерить время: на 1 GPU, на 2 GPU, сравнить ===" << endl;
    
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " <input.png> <output.png> <filter> <mode>" << endl;
        cout << "  filter: blur | edge | denoise" << endl;
        cout << "  mode: single | dual | compare" << endl;
        cout << endl;
        cout << "Examples:" << endl;
        cout << "  " << argv[0] << " input.png output.png edge single" << endl;
        cout << "  " << argv[0] << " input.png output.png blur dual" << endl;
        cout << "  " << argv[0] << " input.png output.png denoise compare" << endl;
        return 1;
    }
    
    const char* input_name = argv[1];
    const char* output_name = argv[2];
    string filter_str = argv[3];
    string mode_str = argv[4];
    
    FilterType filter_type = parse_filter(filter_str);
    ExecutionMode mode = parse_mode(mode_str);
    
    // Check CUDA devices
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 1;
    }
    
    cout << "\nFound " << device_count << " CUDA-capable device(s)" << endl;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "  GPU " << i << ": " << prop.name << endl;
    }
    
    if (mode == MODE_DUAL || mode == MODE_COMPARE) {
        if (device_count < 2) {
            cout << "\n✗ Error: Dual mode requires at least 2 GPUs" << endl;
            cout << "Found only " << device_count << " GPU(s)" << endl;
            return 1;
        }
    }
    
    // Execute based on mode
    switch (mode) {
        case MODE_SINGLE:
            run_single_gpu(input_name, output_name, filter_type);
            break;
            
        case MODE_DUAL:
            run_dual_gpu(input_name, output_name, filter_type);
            break;
            
        case MODE_COMPARE:
            run_comparison(input_name, filter_type);
            break;
    }
    
    cout << "\n=== Program completed successfully ===" << endl;
    return 0;
}