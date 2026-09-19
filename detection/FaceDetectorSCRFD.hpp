#pragma once

#include <opencv2/core.hpp>
#include "hailo/hailort.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>
#include <cstddef>

/**
 * @class FaceDetectorSCRFD
 * @brief SCRFD face detector accelerated by Hailo-8 NPU via HailoRT VStreams.
 *
 * Loads a compiled .hef model onto the NPU and performs real-time face detection on 
 * camera frames. The detected face region is cropped and resized to a fixed 160x208 
 * ROI suitable for downstream rPPG processing.
 */
class FaceDetectorSCRFD {
public:
    // Constants
    // Fixed ROI output dimensions (multiples of 16 for anonimization requirement).
    static constexpr int kRoiW = 160;
    static constexpr int kRoiH = 208;

    // Model input dimensions (SCRFD expects 640x640).
    static constexpr int kInputW = 640;
    static constexpr int kInputH = 640;

    /**
     * @brief Constructs the detector, loads the HEF onto the NPU, and
     *        opens Input/Output VStreams for synchronous inference.
     *
     * @param hef_path Path to the compiled SCRFD .hef file.
     * @param score_threshold Minimum face confidence after sigmoid (default 0.5).
     * @throws std::runtime_error on any HailoRT initialisation failure.
     */
    explicit FaceDetectorSCRFD(const std::string& hef_path, float score_threshold = 0.5f);

    // Destructor - releases NPU resources (VStreams, network, device).
    ~FaceDetectorSCRFD();

    // Non-copyable, non-movable (unique hardware resources).
    FaceDetectorSCRFD(const FaceDetectorSCRFD&)            = delete;
    FaceDetectorSCRFD& operator=(const FaceDetectorSCRFD&) = delete;

    /**
     * @brief Runs face detection on a BGR frame and extracts the face ROI.
     *
     * Pipeline: BGR -> RGB + letterbox -> NPU inference -> decode + NMS ->
     *           coordinate mapping -> crop + resize to 160x208.
     *
     * @param input_bgr Input frame in BGR CV_8UC3.
     * @param[out] out_roi  Face region resized to exactly kRoiW x kRoiH (160x208).
     * @param[out] out_bbox Bounding box of the detected face in original frame coords.
     * @return true if a face was detected and outputs are valid; false otherwise.
     */
    bool processFrame(const cv::Mat& input_bgr, cv::Mat& out_roi, cv::Rect& out_bbox);

private:
    // Internal types 
    // A single decoded detection in the 640x640 letterbox coordinate space.
    struct Detection {
        cv::Rect box;
        float score;
    };

    // Configuration
    float m_score_threshold;
    static constexpr float kNmsIoUThreshold = 0.4f;
    static constexpr int   kNumAnchors      = 2;

    // HailoRT resources
    std::unique_ptr<hailort::VDevice> m_vdevice;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> m_network_group;
    std::vector<hailort::InputVStream> m_input_vstreams;
    std::vector<hailort::OutputVStream> m_output_vstreams;

    // Output buffer management
    // Pre-allocated byte buffers (one per output tensor, sized for FLOAT32).
    std::vector<std::vector<uint8_t>> m_output_buffers;
    /// Maps output tensor name -> index in m_output_vstreams / m_output_buffers.
    std::unordered_map<std::string, size_t> m_output_name_to_idx;

    // Pre-allocated preprocessing buffers
    cv::Mat m_rgb_frame;       // BGR -> RGB conversion result
    cv::Mat m_resized_temp;    // Resized (aspect-preserving) intermediate
    cv::Mat m_letterbox;       // 640x640 padded input for the NPU

    // Letterbox geometry (recomputed if input resolution changes) 
    float m_lb_scale    = 1.0f;
    int   m_lb_pad_left = 0;
    int   m_lb_pad_top  = 0;

    // Private methods
    /**
     * @brief Converts BGR -> RGB and builds a 640x640 letterboxed image.
     */
    void preprocess(const cv::Mat& input_bgr);

    /**
     * @brief Writes the letterboxed image to the NPU and reads all output tensors.
     */
    void runInference();

    /**
     * @brief Decodes all output scales, runs NMS, and maps coords to original frame.
     */
    void postprocess(const cv::Mat& original_frame, std::vector<Detection>& detections);

    /**
     * @brief Decodes detections from one feature-pyramid scale.
     *
     * @param score_data Pointer to dequantised FLOAT32 score tensor (H x W x 2 NHWC).
     * @param bbox_data  Pointer to dequantised FLOAT32 bbox tensor  (H x W x 8 NHWC).
     * @param grid_h     Feature map height.
     * @param grid_w     Feature map width.
     * @param stride     Stride for this scale (8, 16, or 32).
     * @param detections Output vector to append decoded detections to.
     */
    void decodeScale(const float* score_data,
                     const float* bbox_data,
                     int grid_h, int grid_w, int stride,
                     std::vector<Detection>& detections) const;

    /**
     * @brief non-maximum suppression implementation. Keeps the highest-score
     *        detection and suppresses any overlapping box with IoU > iou_threshold.
     */
    static void nms(std::vector<Detection>& detections, float iou_threshold);

    /** @brief Computes Intersection-over-Union between two cv::Rects. */
    static float computeIoU(const cv::Rect& a, const cv::Rect& b);
};
