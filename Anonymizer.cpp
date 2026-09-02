#include "Anonymizer.hpp"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <tuple>

Anonymizer::Anonymizer(int macroblock_size)
    : m_macroblock_size(macroblock_size),
      m_is_calibrated(false),
      m_last_size(0, 0),
      m_last_type(-1)
{
    if (m_macroblock_size <= 0) {
        throw std::invalid_argument("Macroblock size must be greater than 0");
    }
    initializeZigzagLut();
}

void Anonymizer::initializeZigzagLut() {
    int n = m_macroblock_size;
    m_zigzag_lut.resize(n * n);
    
    int row = 0, col = 0;
    bool going_up = true;
    
    for (int i = 0; i < n * n; ++i) {
        m_zigzag_lut[i] = row * n + col;
        
        if (going_up) {
            if (col == n - 1) {
                row++;
                going_up = false;
            } else if (row == 0) {
                col++;
                going_up = false;
            } else {
                row--;
                col++;
            }
        } else {
            if (row == n - 1) {
                col++;
                going_up = true;
            } else if (col == 0) {
                row++;
                going_up = true;
            } else {
                row++;
                col--;
            }
        }
    }
}

void Anonymizer::reset() {
    m_is_calibrated = false;
    m_sigma.clear();
    m_last_size = cv::Size(0, 0);
}

bool Anonymizer::isCalibrated() const {
    return m_is_calibrated;
}

void Anonymizer::calibrate(const cv::Mat& flattened_ycbcr) {
    size_t n_pixels = flattened_ycbcr.rows;
    m_sigma.resize(n_pixels);
    std::iota(m_sigma.begin(), m_sigma.end(), 0);
    
    // cv::cvtColor with COLOR_BGR2YCrCb gives [Y, Cr, Cb] channels
    // The required keys for sorting are: Y primary, Cb secondary, Cr tertiary, in descending order.
    
    const auto* data = flattened_ycbcr.ptr<cv::Vec3b>(0);
    
    // std::stable_sort guarantees O(N log^2 N) time complexity, and keeps equal elements in the same relative order
    std::stable_sort(m_sigma.begin(), m_sigma.end(), [&data](size_t a, size_t b) {
        const auto& pa = data[a];
        const auto& pb = data[b];
        
        // Channel 0: Y
        if (pa[0] != pb[0]) return pa[0] > pb[0];
        
        // Channel 2: Cb
        if (pa[2] != pb[2]) return pa[2] > pb[2];
        
        // Channel 1: Cr
        return pa[1] > pb[1];
    });
    
    m_is_calibrated = true;
}

cv::Mat Anonymizer::processFrame(const cv::Mat& input_roi) {
    // Validation
    if (input_roi.empty()) {
        throw std::invalid_argument("Input ROI is empty");
    }
    if (input_roi.type() != CV_8UC3) {
        throw std::invalid_argument("Input ROI must be of type CV_8UC3");
    }
    if (input_roi.cols % m_macroblock_size != 0 || input_roi.rows % m_macroblock_size != 0) {
        throw std::invalid_argument("Input ROI dimensions must be exact multiples of macroblock size");
    }
    
    // Check if recalibration is needed due to size or type changes
    if (input_roi.size() != m_last_size || input_roi.type() != m_last_type) {
        reset();
        m_last_size = input_roi.size();
        m_last_type = input_roi.type();
    }
    
    // Convert to YCrCb space
    cv::cvtColor(input_roi, m_ycbcr_roi, cv::COLOR_BGR2YCrCb);
    
    // Flatten the YCrCb image to 1D continuous format (n x 1 matrix with 3 channels)
    int n_pixels = m_ycbcr_roi.total();
    m_flattened_ycbcr = m_ycbcr_roi.reshape(3, n_pixels);
    
    // Lexical Sort Calibration (t = 0)
    if (!m_is_calibrated) {
        calibrate(m_flattened_ycbcr);
    }
    
    // Permute pixels according to sigma vector (S[i] = F[sigma[i]])
    m_permuted_pixels.create(n_pixels, 1, CV_8UC3);
    const auto* src_data = m_flattened_ycbcr.ptr<cv::Vec3b>(0);
    auto* dst_data = m_permuted_pixels.ptr<cv::Vec3b>(0);
    
    for (size_t i = 0; i < (size_t)n_pixels; ++i) {
        dst_data[i] = src_data[m_sigma[i]];
    }
    
    // Reorganize into Macroblocks with Zigzag scan
    m_output_roi.create(input_roi.rows, input_roi.cols, CV_8UC3);
    
    int B_x = input_roi.cols / m_macroblock_size;
    int B_y = input_roi.rows / m_macroblock_size;
    int pixels_per_block = m_macroblock_size * m_macroblock_size;
    
    int out_step = m_output_roi.step1() / 3; // Step in pixels
    auto* out_data = m_output_roi.ptr<cv::Vec3b>(0);
    
    size_t permuted_idx = 0;
    
    for (int by = 0; by < B_y; ++by) {
        int block_start_y = by * m_macroblock_size;
        for (int bx = 0; bx < B_x; ++bx) {
            int block_start_x = bx * m_macroblock_size;
            
            // Re-populate the 16x16 macroblock from the permuted sequence 
            // using the zigzag spatial filling curve pattern
            for (int k = 0; k < pixels_per_block; ++k) {
                int zigzag_offset = m_zigzag_lut[k];
                int local_y = zigzag_offset / m_macroblock_size;
                int local_x = zigzag_offset % m_macroblock_size;
                
                int global_y = block_start_y + local_y;
                int global_x = block_start_x + local_x;
                
                out_data[global_y * out_step + global_x] = dst_data[permuted_idx++];
            }
        }
    }
    
    // Convert back to original color space (BGR)
    cv::Mat final_result;
    cv::cvtColor(m_output_roi, final_result, cv::COLOR_YCrCb2BGR);
    
    return final_result;
}
