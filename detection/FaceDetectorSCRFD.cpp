#include "FaceDetectorSCRFD.hpp"
#include "DetectionUtils.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

// Describes one feature-pyramid scale of the SCRFD-2.5G HEF output.
struct ScaleInfo {
    const char* score_name;   // Classification logits  (H * W * 2)
    const char* bbox_name;    // Bounding-box distances (H * W * 8)
    int stride;               // Spatial stride (8, 16, or 32)
    int grid_h;               // Feature-map height
    int grid_w;               // Feature-map width
};

// The three scales of SCRFD-2.5G.
static const ScaleInfo kScales[] = {
    {"scrfd_2_5g/conv42", "scrfd_2_5g/conv43",  8, 80, 80},
    {"scrfd_2_5g/conv49", "scrfd_2_5g/conv50", 16, 40, 40},
    {"scrfd_2_5g/conv55", "scrfd_2_5g/conv56", 32, 20, 20},
};

}

// Constructor - HailoRT initialisation
FaceDetectorSCRFD::FaceDetectorSCRFD(const std::string& hef_path, float score_threshold)
    : m_score_threshold(score_threshold)
{
    using namespace hailort;

    // Create VDevice
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        throw std::runtime_error(
            "Failed to create Hailo VDevice (status=" + std::to_string(static_cast<int>(vdevice_exp.status())) + ")."
            " Is the Hailo-8 device connected and hailort driver loaded?");
    }
    m_vdevice = vdevice_exp.release();

    // Load compiled model (.hef)
    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) {
        throw std::runtime_error("Failed to load HEF file: " + hef_path);
    }
    auto hef = hef_exp.release();

    // Configure network group on the device
    auto cfg_params_exp = m_vdevice->create_configure_params(hef);
    if (!cfg_params_exp) {
        throw std::runtime_error("Failed to create network configure params");
    }
    auto net_groups_exp = m_vdevice->configure(hef, cfg_params_exp.value());
    if (!net_groups_exp) {
        throw std::runtime_error("Failed to configure network group on device (status=" + 
                                std::to_string(static_cast<int>(net_groups_exp.status())) + ")");
    }
    m_network_group = std::move(net_groups_exp.value()[0]);

    // Input VStream params (UINT8 RGB, auto quantisation)
    auto in_params_exp = m_network_group->make_input_vstream_params(
        true,  // quantized = true (we send raw uint8)
        HAILO_FORMAT_TYPE_AUTO,  // match HEF native input format
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        1);  // queue_size = 1 (low latency)
    if (!in_params_exp) {
        throw std::runtime_error("Failed to create input VStream params");
    }

    // Output VStream params (dequantised FLOAT32, NHWC)
    auto out_params_exp = m_network_group->make_output_vstream_params(
        false,  // quantized = false -> dequantise
        HAILO_FORMAT_TYPE_FLOAT32,  // output as 32-bit float
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS,
        1);  // queue_size = 1 (low latency)
    if (!out_params_exp) {
        throw std::runtime_error("Failed to create output VStream params");
    }

    // Build VStreams
    auto in_vs_exp = VStreamsBuilder::create_input_vstreams(*m_network_group, in_params_exp.value());
    if (!in_vs_exp) {
        throw std::runtime_error("Failed to create input VStreams");
    }
    m_input_vstreams = in_vs_exp.release();

    auto out_vs_exp = VStreamsBuilder::create_output_vstreams(*m_network_group, out_params_exp.value());
    if (!out_vs_exp) {
        throw std::runtime_error("Failed to create output VStreams");
    }
    m_output_vstreams = out_vs_exp.release();

    // Pre-allocate output buffers and build name -> index map
    m_output_buffers.resize(m_output_vstreams.size());
    for (size_t i = 0; i < m_output_vstreams.size(); ++i) {
        m_output_buffers[i].resize(m_output_vstreams[i].get_frame_size());
        m_output_name_to_idx[m_output_vstreams[i].name()] = i;
    }

    // Pre-allocate the 640x640 letterbox buffer (black padding)
    m_letterbox = cv::Mat::zeros(kInputH, kInputW, CV_8UC3);
}

// Destructor
FaceDetectorSCRFD::~FaceDetectorSCRFD()
{
    // Destruction order (enforced by member declaration order):
    //   1. m_output_vstreams / m_input_vstreams -> close virtual streams.
    //   2. m_network_group (shared_ptr) -> releases configuration.
    //   3. m_vdevice -> releases the PCIe device handle.
    // All managed by smart pointers no manual cleanup needed.
}

bool FaceDetectorSCRFD::processFrame(const cv::Mat& input_bgr, cv::Mat& out_roi, cv::Rect& out_bbox)
{
    if (input_bgr.empty() || input_bgr.type() != CV_8UC3) {
        return false;
    }

    // BGR -> RGB, letterbox to 640x640
    preprocess(input_bgr);
    // Write to NPU, read output tensors
    runInference();
    // Decode detections from all 3 scales, apply NMS, map coordinates
    std::vector<Detection> detections;
    detections.reserve(128);
    postprocess(input_bgr, detections);

    if (detections.empty()) {
        return false;
    }

    // Select the best detection (highest score, index 0 after NMS sort)
    const Detection& best = detections[0];
    out_bbox = best.box;

    // Safety-clamp to frame boundaries
    const int x = std::max(0, out_bbox.x);
    const int y = std::max(0, out_bbox.y);
    const int w = std::min(out_bbox.width,  input_bgr.cols - x);
    const int h = std::min(out_bbox.height, input_bgr.rows - y);
    if (w <= 0 || h <= 0) {
        return false;
    }
    out_bbox = cv::Rect(x, y, w, h);

    // Crop face ROI (zero-copy sub-matrix) -> resize to fixed 160x208
    cv::resize(input_bgr(out_bbox), out_roi, cv::Size(kRoiW, kRoiH), 0, 0, cv::INTER_LINEAR);

    return true;
}

// Preprocessing: BGR -> RGB + letterbox to 640x640
void FaceDetectorSCRFD::preprocess(const cv::Mat& input_bgr)
{
    // Convert BGR -> RGB (SCRFD expects RGB input)
    cv::cvtColor(input_bgr, m_rgb_frame, cv::COLOR_BGR2RGB);

    // Compute letterbox scaling (preserve aspect ratio, pad remainder)
    computeLetterboxParams(input_bgr.cols, input_bgr.rows, kInputW, m_lb_scale, m_lb_pad_left, m_lb_pad_top);

    const int new_w = static_cast<int>(input_bgr.cols * m_lb_scale);
    const int new_h = static_cast<int>(input_bgr.rows * m_lb_scale);

    // Resize maintaining aspect ratio
    cv::resize(m_rgb_frame, m_resized_temp, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // Black-fill the 640x640 canvas, then copy the resized image to its centre
    m_letterbox.setTo(cv::Scalar(0, 0, 0));
    m_resized_temp.copyTo(m_letterbox(cv::Rect(m_lb_pad_left, m_lb_pad_top, new_w, new_h)));

    // Release intermediate buffers immediately to free ARM memory
    m_rgb_frame.release();
    m_resized_temp.release();
}

// NPU inference via HailoRT VStreams
void FaceDetectorSCRFD::runInference()
{
    // Write the 640x640x3 RGB letterboxed image to the NPU input queue.
    const size_t input_size = static_cast<size_t>(kInputW) * kInputH * 3;
    hailo_status status = m_input_vstreams[0].write(hailort::MemoryView(m_letterbox.data, input_size));
    if (HAILO_SUCCESS != status) {
        throw std::runtime_error(
            "NPU input write failed (status=" + std::to_string(static_cast<int>(status)) + ")");
    }

    // Read results from every output VStream.
    for (size_t i = 0; i < m_output_vstreams.size(); ++i) {
        status = m_output_vstreams[i].read(hailort::MemoryView(m_output_buffers[i].data(), m_output_buffers[i].size()));
        if (HAILO_SUCCESS != status) {
            throw std::runtime_error(
                "NPU output read failed for '" + m_output_vstreams[i].name() + "' (status=" +
                std::to_string(static_cast<int>(status)) + ")");
        }
    }
}

// Post-processing: decode 3 scales -> NMS -> coordinate mapping
void FaceDetectorSCRFD::postprocess(const cv::Mat& original_frame, std::vector<Detection>& detections)
{
    detections.clear();

    // Decode detections from each feature-pyramid scale
    static bool first_call = true;
    for (const auto& s : kScales) {
        auto score_it = m_output_name_to_idx.find(s.score_name);
        auto bbox_it  = m_output_name_to_idx.find(s.bbox_name);

        if (score_it == m_output_name_to_idx.end() || bbox_it == m_output_name_to_idx.end()) {
            throw std::runtime_error(
                std::string("Expected output tensor not found in HEF: ") +
                s.score_name + " or " + s.bbox_name + ". Check that the HEF matches the SCRFD architecture.");
        }

        const auto* score_data = reinterpret_cast<const float*>(m_output_buffers[score_it->second].data());
        const auto* bbox_data  = reinterpret_cast<const float*>(m_output_buffers[bbox_it->second].data());

        decodeScale(score_data, bbox_data, s.grid_h, s.grid_w, s.stride, detections);
    }

    // Non-Maximum Suppression
    ::nms(detections, kNmsIoUThreshold);

    // Map from 640x640 letterbox space -> original frame coordinates
    const int orig_w = original_frame.cols;
    const int orig_h = original_frame.rows;

    for (auto& det : detections) {
        const float x = (static_cast<float>(det.box.x) - m_lb_pad_left) / m_lb_scale;
        const float y = (static_cast<float>(det.box.y) - m_lb_pad_top)  / m_lb_scale;
        const float w = static_cast<float>(det.box.width)  / m_lb_scale;
        const float h = static_cast<float>(det.box.height) / m_lb_scale;

        const int xi = std::max(0, static_cast<int>(std::round(x)));
        const int yi = std::max(0, static_cast<int>(std::round(y)));
        const int wi = std::min(static_cast<int>(std::round(w)), orig_w - xi);
        const int hi = std::min(static_cast<int>(std::round(h)), orig_h - yi);

        det.box = cv::Rect(xi, yi, std::max(1, wi), std::max(1, hi));
    }
}

// decodeScale — decode one feature-pyramid level
void FaceDetectorSCRFD::decodeScale(
    const float* score_data,
    const float* bbox_data,
    int grid_h, int grid_w, int stride,
    std::vector<Detection>& detections) const
{
    for (int row = 0; row < grid_h; ++row) {
        for (int col = 0; col < grid_w; ++col) {
            for (int a = 0; a < kNumAnchors; ++a) {

                // Score
                // NHWC layout: data[(row * W + col) * C + c]
                // Score tensor has C=2 (one score per anchor).
                const int score_idx = (row * grid_w + col) * kNumAnchors + a;

                // The HEF outputs are already probabilities (sigmoid is baked into the model).
                const float score = score_data[score_idx];

                if (score < m_score_threshold) {
                    continue;
                }

                // Anchor centre (640x640 letterbox space)
                const float cx = (static_cast<float>(col) + 0.5f) * stride;
                const float cy = (static_cast<float>(row) + 0.5f) * stride;

                // Bounding-box distances
                const int bbox_base = (row * grid_w + col) * (kNumAnchors * 4) + a * 4;
                const float dl = bbox_data[bbox_base + 0] * stride;
                const float dt = bbox_data[bbox_base + 1] * stride;
                const float dr = bbox_data[bbox_base + 2] * stride;
                const float db = bbox_data[bbox_base + 3] * stride;

                float x1 = cx - dl;
                float y1 = cy - dt;
                float x2 = cx + dr;
                float y2 = cy + db;

                // Clamp to the 640x640 input boundaries
                x1 = std::max(0.0f, std::min(x1, static_cast<float>(kInputW)));
                y1 = std::max(0.0f, std::min(y1, static_cast<float>(kInputH)));
                x2 = std::max(0.0f, std::min(x2, static_cast<float>(kInputW)));
                y2 = std::max(0.0f, std::min(y2, static_cast<float>(kInputH)));

                const int bw = static_cast<int>(x2 - x1);
                const int bh = static_cast<int>(y2 - y1);
                if (bw <= 0 || bh <= 0) {
                    continue;
                }

                detections.push_back({cv::Rect(static_cast<int>(x1), static_cast<int>(y1), bw, bh), score});
            }
        }
    }
}

