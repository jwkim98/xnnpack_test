#include <xnnpack.h>
#include <array>
#include <cpp-oss-template/Test.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

int TestXnn() {
    // Define input dimensions and other parameters
    size_t batchSize = 1;
    size_t inputHeight = 96;
    size_t inputWidth = 96;
    size_t inputChannels = 32;
    size_t outputChannels = 8;
    size_t kernelSize = 1;
    size_t stride = 1;
    size_t padding = 0;
    size_t dilationWidth = 1;
    size_t dilationHeight = 1;
    size_t outputHeight = (inputHeight + padding + padding - dilationHeight * (kernelSize - 1) - 1) / stride + 1;
    size_t outputWidth = (inputWidth + padding + padding - dilationWidth * (kernelSize - 1) - 1) / stride + 1;

    const std::size_t inputSize = batchSize * inputHeight * inputWidth * inputChannels;
    const std::size_t outputSize = batchSize * outputHeight * outputWidth * outputChannels;

    // Allocate memory for input and output tensors
    auto *inputData = (float *) std::aligned_alloc(64, sizeof(float) * inputSize + XNN_EXTRA_BYTES);
    auto *outputData = (float *) std::aligned_alloc(64, sizeof(float) * outputSize + XNN_EXTRA_BYTES);

    // Create XNNPACK convolution operators
    xnn_operator_t convOp = nullptr;

    xnn_code_cache *codeCachePtr = nullptr;

    xnn_status status = xnn_initialize(nullptr);

//     xnn_weights_cache_t* weightCache = nullptr;
//     status = xnn_create_weights_cache(weightCache);
//
//    if (status != xnn_status_success) {
//        std::cerr << "Error creating weights cache." << std::endl;
//        return 1;
//    }

    std::vector<float> weights(
            outputChannels * kernelSize * kernelSize * inputChannels, 1.0f);

    std::vector<float> bias(outputChannels, 1.0f);

    status = xnn_create_convolution2d_nhwc_f32(
            padding /* top padding */, padding /* right padding */, padding /* bottom padding */,
            padding /* left padding */, kernelSize /* kernel height */, kernelSize /* kernel width */,
            stride /* subsampling height */, stride /* subsampling width */,
            dilationHeight/* dilation_height */, dilationWidth /* dilation_width */, 1 /* groups */,
            inputChannels /* input channels per group */, outputChannels /* output_channels_per_group */,
            inputChannels /* input pixel stride */, outputChannels /* output pixel stride */,
            weights.data(), bias.data(), 0.0f /* output min */,
            6.0f /* output max */, 0 /* flags */, codeCachePtr, nullptr, &convOp);

    if (status != xnn_status_success) {
        std::cout << "status: " << status << std::endl;
        std::cerr << "Error creating XNNPACK convolution operator."
                  << std::endl;
        xnn_delete_operator(convOp);
        return 1;
    }

    pthreadpool_t threadPool = pthreadpool_create(1);
    status = xnn_reshape_convolution2d_nhwc_f32(
            convOp,
            /*batch_size=*/1, /*input_height=*/inputHeight, /*input_width=*/inputWidth,
            /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
            /*threadpool=*/threadPool);

    if (status != xnn_status_success) {
        std::cout << "status: " << status << std::endl;
        std::cerr << "Error reshaping convolution operator." << std::endl;
        xnn_delete_operator(convOp);
        return 1;
    }

    // Set up convolution operator 2
    status =
            xnn_setup_convolution2d_nhwc_f32(convOp, inputData, outputData);

    if (status != xnn_status_success) {
        std::cout << "status: " << status << std::endl;

        std::cerr << "Error setting up XNNPACK convolution operator."
                  << std::endl;
        xnn_delete_operator(convOp);
        return 1;
    }

    int iterations = 1000;
    int unitIterations = 10;
    std::vector<long> times(iterations / unitIterations);

    for (int i = 0; i < iterations / unitIterations; ++i) {
        auto begin = std::chrono::system_clock::now();
        for (int j = 0; j < unitIterations; ++j) {
            status = xnn_run_operator(convOp, nullptr);
        }
        auto end = std::chrono::system_clock::now();
        times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / unitIterations;
    }

    if (status != xnn_status_success) {
        std::cerr << "Error running XNNPACK convolution operator (Layer 2)."
                  << std::endl;
        xnn_delete_operator(convOp);
        return 1;
    }

    std::cout << "Total Average : " << std::accumulate(times.begin(), times.end(), 1L) / static_cast<long>(times.size())
              << "us" << std::endl;
    std::cout << "Average without first (" << unitIterations * 3 << ") iterations : "
              << std::accumulate(times.begin() + 3, times.end(), 1L) / static_cast<long>(times.size() - 3) << "us"
              << std::endl;

    std::sort(times.begin(), times.end());
    std::cout << "Slowest iteration : (" << times.back() << ")" << std::endl;
    std::cout << "Fastest iteration : (" << *times.begin() << ")" << std::endl;

    // Access the output tensor values (dummy example: print the first 10
    // values)
    for (size_t i = 0;
         i < std::min(outputSize, static_cast<size_t>(10)); ++i) {
        std::cout << "Output value " << i << outputData[i] << std::endl;
    }

    // Clean up resources
    xnn_delete_operator(convOp);
    free(inputData);
    free(outputData);

    return 0;
}

int Add(int a, int b) {
    return a + b;
}
