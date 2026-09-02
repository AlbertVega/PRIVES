#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>

class Anonymizer {
public:
    /**
     * @brief Constructor for Anonymizer.
     * @param macroblock_size The size of the macroblock (default 16).
     */
    explicit Anonymizer(int macroblock_size = 16);
    
    /**
     * @brief Processes a video frame/ROI applying the spatial redistribution anonymization.
     * @param input_roi The input Region of Interest (ROI) image, CV_8UC3 format.
     * @return The anonymized ROI image.
     */
    cv::Mat processFrame(const cv::Mat& input_roi);
    
    /**
     * @brief Resets the calibration state, useful when face tracking is lost or ROI changes.
     */
    void reset();
    
    /**
     * @brief Checks if the Anonymizer is calibrated (t > 0).
     * @return true if calibrated, false otherwise.
     */
    bool isCalibrated() const;

private:
    int m_macroblock_size;
    bool m_is_calibrated;
    cv::Size m_last_size;
    int m_last_type;
    
    // Pre-allocated buffers to avoid heap allocations in the main processing loop
    std::vector<size_t> m_sigma;
    cv::Mat m_ycbcr_roi;
    cv::Mat m_flattened_ycbcr;
    cv::Mat m_permuted_pixels;
    cv::Mat m_output_roi;
    
    // Zigzag lookup table (LUT) for macroblock mapping
    // Maps a 1D sorted index to a 1D offset in the macroblock (y * size + x)
    std::vector<int> m_zigzag_lut;

    void initializeZigzagLut();
    void calibrate(const cv::Mat& flattened_ycbcr);
};
