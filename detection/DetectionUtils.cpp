#include "DetectionUtils.hpp"
#include <algorithm>
#include <cmath>

void nms(std::vector<Detection>& detections, float iou_threshold)
{
    if (detections.empty()) {
        return;
    }

    // Sort by score descending - the first surviving detection is the best.
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> kept;
    kept.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        kept.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            if (computeIoU(detections[i].box, detections[j].box) >
                iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    detections = std::move(kept);
}

float computeIoU(const cv::Rect& a, const cv::Rect& b)
{
    const cv::Rect intersection = a & b;
    const float area_inter = static_cast<float>(intersection.area());
    const float area_union = static_cast<float>(a.area() + b.area()) - area_inter;
    return (area_union > 0.0f) ? (area_inter / area_union) : 0.0f;
}

void computeLetterboxParams(int input_w, int input_h, int target_size, float& scale, int& pad_left, int& pad_top)
{
    const float scale_w = static_cast<float>(target_size) / input_w;
    const float scale_h = static_cast<float>(target_size) / input_h;
    scale = std::min(scale_w, scale_h);

    const int new_w = static_cast<int>(input_w * scale);
    const int new_h = static_cast<int>(input_h * scale);

    pad_left = (target_size - new_w) / 2;
    pad_top  = (target_size - new_h) / 2;
}
