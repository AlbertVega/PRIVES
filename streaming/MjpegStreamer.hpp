#pragma once

/**
 * @file MjpegStreamer.hpp
 * @brief MJPEG-over-HTTP streaming server with pipeline overlay composition.
 *
 * Serves a live MJPEG stream on a configurable TCP port. A background thread
 * accepts one client at a time and pushes JPEG frames separated by standard
 * multipart MJPEG boundary markers.
 */

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ----------------------------------------------------------------------------
// FrameInfo - metadata bundle for overlay composition
// ----------------------------------------------------------------------------

/**
 * @brief Lightweight POD that carries per-frame pipeline state into the
 *        streamer so it can render a complete HUD overlay.
 */
struct FrameInfo {
    bool     detected     = false;   // True if a face was found this frame.
    cv::Rect bbox;                   // Face bounding box in original frame coords.
    cv::Mat  roi;                    // 160*208 cropped face ROI (empty when !detected).
    double   latency_ms   = 0.0;     // Detection latency for this frame (ms).
    double   fps          = 0.0;     // Running average frames-per-second.
    double   hr_bpm       = 0.0;     // Latest heart-rate estimate (BPM).
    double   fr_rpm       = 0.0;     // Latest respiratory-rate estimate (RPM).
    bool     vitals_ready = false;   // True once the estimator has calibrated.
};

// ----------------------------------------------------------------------------
// MjpegStreamer
// ----------------------------------------------------------------------------

class MjpegStreamer {
public:
    /**
     * @brief Creates the TCP server socket and starts the background accept/send thread.
     *
     * @param port         TCP port to listen on (default 8080).
     * @param jpeg_quality JPEG encoding quality 0-100 (default 50).
     * @throws std::runtime_error on socket/bind/listen failure.
     */
    explicit MjpegStreamer(int port = 8080, int jpeg_quality = 50);

    /**
     * @brief Shuts down the server socket, wakes any blocked threads, and joins the background thread.
     */
    ~MjpegStreamer();

    // Non-copyable, non-movable (owns a socket + thread).
    MjpegStreamer(const MjpegStreamer&)            = delete;
    MjpegStreamer& operator=(const MjpegStreamer&) = delete;

    /**
     * @brief Composes a HUD overlay onto @p frame and queues the result for MJPEG streaming.
     *
     * The overlay includes:
     *   - Green bounding box + "Face" label (when detected).
     *   - 160*208 ROI inset in the top-right corner (when detected).
     *   - "No face detected" warning (when not detected).
     *   - FPS and detection latency readout (bottom-left).
     *   - HR / FR vital signs readout (bottom-left, below FPS).
     *
     * The method works on a shallow copy of @p frame internally, so the
     * caller may release the original immediately after this call returns.
     *
     * @param frame The raw camera frame (BGR, CV_8UC3).
     * @param info  Per-frame pipeline metadata (see FrameInfo).
     */
    void pushFrame(const cv::Mat& frame, const FrameInfo& info);

private:
    /**
     * @brief Background thread: accepts one TCP client at a time and streams JPEG frames with MJPEG multipart boundaries.
     */
    void serverLoop();

    /**
     * @brief Draws all HUD elements onto @p canvas according to @p info.
     */
    static void composeOverlay(cv::Mat& canvas, const FrameInfo& info);

    /**
     * @brief Sends all bytes on @p fd, retrying on partial writes.
     * @return Total bytes sent, or -1 on error.
     */
    static ssize_t sendAll(int fd, const void* data, size_t len);

    // --- Members ---
    int                     m_server_fd    = -1;
    int                     m_port;
    int                     m_jpeg_quality;
    std::atomic<bool>       m_running{true};
    std::thread             m_thread;
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    std::vector<uint8_t>    m_jpeg_buffer;
    bool                    m_has_frame    = false;
};
