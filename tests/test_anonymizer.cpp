#include "../anonymizer/Anonymizer.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <filesystem>

int main(int argc, char* argv[]) {
    std::cout << "Starting Test for Anonymizer..." << std::endl;

    // load an image to be anonymized (must be in the same directory as the executable or a valid path must be provided as an argument)
    const std::string img_path = (argc > 1) ? argv[1] : "face.jpg";
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "ERROR: Could not load the image: " << img_path << std::endl;
        std::cerr << "Usage: test_anonymizer [image_path]" << std::endl;
        return 1;
    }
    std::cout << "Image loaded: " << img_path
              << " (" << img.cols << "x" << img.rows << ")" << std::endl;

    Anonymizer anonymizer(16);

    // Calculate the mean of the original image
    cv::Scalar mean_orig = cv::mean(img);

    // Process for t=0 (Calibration and Permutation)
    cv::Mat anonymized_t0 = anonymizer.processFrame(img);

    // Calculate mean of t=0
    cv::Scalar mean_t0 = cv::mean(anonymized_t0);

    // Process for t=1 (Second pass with the same real image)
    cv::Mat img_t1 = img.clone();
    cv::Scalar mean_orig_t1 = cv::mean(img_t1);
    cv::Mat anonymized_t1 = anonymizer.processFrame(img_t1);
    cv::Scalar mean_t1 = cv::mean(anonymized_t1);

    // Save results to disk
    const std::filesystem::path output_dir = "anonymizer_results";
    std::filesystem::create_directories(output_dir);

    cv::imwrite((output_dir / "original.png").string(), img);
    cv::imwrite((output_dir / "anonymized_t0.png").string(), anonymized_t0);
    cv::imwrite((output_dir / "anonymized_t1.png").string(), anonymized_t1);
    std::cout << "Images saved in build/" << output_dir.string() << " directory" << std::endl;

    // Spatial invariance verification
    // MAE < 10^-4 is expected, but due to rounding in integer conversion RGB <-> YCbCr 
    // OpenCV generates a slight numerical error. We tolerate up to 1.0 for 8-bit colors.
    double max_err_t0 = 0.0;
    for (int c = 0; c < 3; ++c) {
        max_err_t0 = std::max(max_err_t0, std::abs(mean_orig[c] - mean_t0[c]));
    }
    
    double max_err_t1 = 0.0;
    for (int c = 0; c < 3; ++c) {
        max_err_t1 = std::max(max_err_t1, std::abs(mean_orig_t1[c] - mean_t1[c]));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Original Mean t=0: [" << mean_orig[0] << ", " << mean_orig[1] << ", " << mean_orig[2] << "]" << std::endl;
    std::cout << "Anonymized Mean t=0: [" << mean_t0[0] << ", " << mean_t0[1] << ", " << mean_t0[2] << "]" << std::endl;
    std::cout << "Max MAE at t=0: " << max_err_t0 << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Original Mean t=1: [" << mean_orig_t1[0] << ", " << mean_orig_t1[1] << ", " << mean_orig_t1[2] << "]" << std::endl;
    std::cout << "Anonymized Mean t=1: [" << mean_t1[0] << ", " << mean_t1[1] << ", " << mean_t1[2] << "]" << std::endl;
    std::cout << "Max MAE at t=1: " << max_err_t1 << std::endl;

    double tolerance = 1.0; // Tolerance due to integer rounding of the conversion YCrCb to BGR
    
    if (max_err_t0 > tolerance) {
        std::cerr << "FAIL: Mean difference too high at t=0. Expected <= " << tolerance << ", got " << max_err_t0 << std::endl;
        return 1;
    }
    
    if (max_err_t1 > tolerance) {
        std::cerr << "FAIL: Mean difference too high at t=1. Expected <= " << tolerance << ", got " << max_err_t1 << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: Mathematical invariance verification passed." << std::endl;
    return 0;
}
