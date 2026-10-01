#pragma once

#include <opencv2/core.hpp>
#include <vector>

/**
 * @brief A single decoded face detection (box + confidence score).
 *
 * This struct is shared between the hardware-accelerated FaceDetectorSCRFD
 * class and the pure-algorithmic post-processing utilities below
 */
struct Detection {
    cv::Rect box;
    float score;
};

/**
 * @brief Non-Maximum Suppression.
 *
 * Sorts detections by score (descending) and greedily suppresses any
 * detection whose IoU with a higher-scoring detection exceeds the threshold.
 *
 * @param detections  In/out vector of detections; suppressed entries are removed.
 * @param iou_threshold  IoU above which a lower-score detection is suppressed.
 */
void nms(std::vector<Detection>& detections, float iou_threshold);

/**
 * @brief Intersection-over-Union between two axis-aligned rectangles.
 * @return IoU in [0, 1], or 0 if the union area is zero.
 */
float computeIoU(const cv::Rect& a, const cv::Rect& b);

/**
 * @brief Computes letterbox scaling parameters for a given input size
 *        mapped onto a square target (target_size x target_size).
 *
 * @param input_w       Width of the original frame.
 * @param input_h       Height of the original frame.
 * @param target_size   Side length of the square output (e.g. 640).
 * @param[out] scale    Uniform scale factor applied to the input.
 * @param[out] pad_left Horizontal padding (pixels) added on the left.
 * @param[out] pad_top  Vertical padding (pixels) added on the top.
 */
void computeLetterboxParams(int input_w, int input_h, int target_size, float& scale, int& pad_left, int& pad_top);
