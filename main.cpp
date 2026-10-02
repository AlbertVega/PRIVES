/**
 * @file main.cpp
 * @brief PRIVES Pipeline Orchestrator.
 *
 * Three-thread architecture:
 *   Thread 1 (Capture)   : GStreamer v4l2src  ->  capture_queue
 *   Thread 2 (Detection) : capture_queue  ->  FaceDetectorSCRFD  ->  roi_queue
 *   Thread 3 (DSP)       : roi_queue  ->  Anonymizer  ->  PosExtractor  ->  estimation
 *
 * Build requirements: OpenCV 4.x, HailoRT, GStreamer 1.x, FFTW3, Iir1, C++17.
 */

#include "detection/FaceDetectorSCRFD.hpp"
#include "anonymizer/Anonymizer.hpp"
#include "extraction/PosExtractor.hpp"
#include "estimation/estimation.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <csignal>
#include <optional>
#include <cstdlib>

// ----------------------------------------------------------------------------
// Global shutdown flag
// ----------------------------------------------------------------------------

static std::atomic<bool> g_running{true};

static void signalHandler(int /*signum*/)
{
    g_running.store(false, std::memory_order_relaxed);
}

// ----------------------------------------------------------------------------
// Thread-safe bounded queue
// ----------------------------------------------------------------------------

/**
 * @brief A bounded, thread-safe single-producer/single-consumer queue.
 *
 * When the queue is full the oldest element is silently discarded to prevent
 * buffer bloat - the consumer always works on the freshest data available.
 *
 * @tparam T Element type (must be move-constructible).
 */
template <typename T>
class BoundedQueue
{
public:
    /**
     * @param capacity Maximum number of elements before old items are dropped.
     */
    explicit BoundedQueue(std::size_t capacity) : m_capacity(capacity)
    {
        m_buffer.reserve(capacity);
    }

    /**
     * @brief Push an element into the queue.
     *
     * If the queue is already at capacity the front (oldest) element is
     * discarded so the consumer never processes stale data.
     *
     * @param item Element to enqueue (moved in).
     */
    void push(T item)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_buffer.size() >= m_capacity) {
                // Drop the oldest element (front) to avoid latency build-up.
                m_buffer.erase(m_buffer.begin());
            }
            m_buffer.push_back(std::move(item));
        }
        m_cv.notify_one();
    }

    /**
     * @brief Pop the front element, blocking until one is available.
     *
     * Returns std::nullopt if the pipeline is shutting down and the queue is
     * empty, so consumer threads can exit cleanly.
     *
     * @param running Reference to the global shutdown flag.
     * @return The dequeued element, or std::nullopt on shutdown.
     */
    std::optional<T> pop(const std::atomic<bool>& running)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&] {
            return !m_buffer.empty() || !running.load(std::memory_order_relaxed);
        });

        if (m_buffer.empty()) {
            return std::nullopt;  // Shutting down.
        }

        T item = std::move(m_buffer.front());
        m_buffer.erase(m_buffer.begin());
        return item;
    }

    /**
     * @brief Wake up any thread blocked in pop() so it can observe the shutdown flag.
     */
    void shutdown()
    {
        m_cv.notify_all();
    }

private:
    std::size_t              m_capacity;
    std::vector<T>           m_buffer;
    std::mutex               m_mutex;
    std::condition_variable  m_cv;
};

// ----------------------------------------------------------------------------
// Pipeline configuration
// ----------------------------------------------------------------------------

namespace cfg {
    // GStreamer capture parameters
    constexpr int    kCaptureWidth   = 640;
    constexpr int    kCaptureHeight  = 480;
    constexpr int    kCaptureFps     = 30;
    constexpr char   kCameraDevice[] = "/dev/video0";

    // Queue capacities (kept tiny to minimise latency)
    constexpr std::size_t kCaptureQueueSize = 2;
    constexpr std::size_t kRoiQueueSize     = 1;

    // SCRFD model
    constexpr char  kHefPath[]        = "scrfd_2.5g.hef";
    constexpr float kScoreThreshold   = 0.5f;
}

// ----------------------------------------------------------------------------
// Thread 1 - Capture (Producer)
// ----------------------------------------------------------------------------

/**
 * @brief Continuously grabs frames from a GStreamer v4l2 pipeline and pushes them into the detection queue.
 *
 * The GStreamer sink is configured with `drop=true max-buffers=1` so the kernel
 * driver never stalls if the consumer is slower than the camera.
 */
static void captureThread(BoundedQueue<cv::Mat>& capture_queue)
{
    const std::string gst_pipeline =
        "v4l2src device=" + std::string(cfg::kCameraDevice) + " ! "
        "video/x-raw,width="  + std::to_string(cfg::kCaptureWidth)  + ","
        "height=" + std::to_string(cfg::kCaptureHeight) + ","
        "framerate=" + std::to_string(cfg::kCaptureFps) + "/1 ! "
        "videoconvert ! "
        "appsink drop=true max-buffers=1";

    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "[CAPTURE] ERROR: Failed to open GStreamer pipeline:\n"
                  << "  " << gst_pipeline << "\n"
                  << "  Ensure the camera is connected at " << cfg::kCameraDevice
                  << " and GStreamer v4l2 plugins are installed." << std::endl;
        g_running.store(false);
        capture_queue.shutdown();
        return;
    }

    std::cout << "[CAPTURE] GStreamer pipeline opened ("
              << cfg::kCaptureWidth << "x" << cfg::kCaptureHeight
              << " @ " << cfg::kCaptureFps << " FPS)." << std::endl;

    cv::Mat frame;

    while (g_running.load(std::memory_order_relaxed)) {
        if (!cap.read(frame) || frame.empty()) {
            // Transient read failure (USB hiccup, etc.) - retry silently.
            continue;
        }
        // Push a *copy* so the capture buffer can be reused immediately.
        capture_queue.push(frame.clone());
    }

    cap.release();
    capture_queue.shutdown();
    std::cout << "[CAPTURE] Thread exiting." << std::endl;
}

// ----------------------------------------------------------------------------
// Thread 2 - Detection (Consumer / Producer)
// ----------------------------------------------------------------------------

/**
 * @brief Runs SCRFD face detection on each captured frame.
 *
 * On detection the 160*208 ROI is forwarded to the DSP queue. The original
 * camera frame is released immediately after detection
 *
 * When tracking is lost (no detection), all downstream modules are reset so
 * that signal buffers do not mix data from different subjects.
 *
 * @param capture_queue Incoming frames from the capture thread.
 * @param roi_queue     Outgoing 160*208 ROIs to the DSP thread.
 * @param anonymizer    Shared reference for cross-thread reset.
 * @param pos_extractor Shared reference for cross-thread reset.
 * @param estimator     Shared reference for cross-thread reset.
 * @param dsp_mutex     Protects reset() calls on DSP-owned objects.
 */
static void detectionThread(BoundedQueue<cv::Mat>& capture_queue,
                            BoundedQueue<cv::Mat>& roi_queue,
                            Anonymizer&            anonymizer,
                            PosExtractor&          pos_extractor,
                            estimation&            estimator,
                            std::mutex&            dsp_mutex)
{
    // Initialise the NPU-accelerated detector.
    FaceDetectorSCRFD detector(cfg::kHefPath, cfg::kScoreThreshold);
    std::cout << "[DETECT]  SCRFD loaded on Hailo-8 NPU." << std::endl;

    cv::Mat  roi;
    cv::Rect bbox;
    bool     was_tracking = false;

    while (g_running.load(std::memory_order_relaxed)) {
        auto maybe_frame = capture_queue.pop(g_running);
        if (!maybe_frame.has_value()) {
            break;  // Shutdown.
        }

        cv::Mat frame = std::move(maybe_frame.value());

        const bool detected = detector.processFrame(frame, roi, bbox);

        // Destroy the original frame immediately.
        frame.release();

        if (detected) {
            was_tracking = true;
            // Forward the ROI (clone to decouple from detector internals).
            roi_queue.push(roi.clone());
        } else {
            if (was_tracking) {
                // Tracking was just lost - reset all downstream state.
                std::lock_guard<std::mutex> lock(dsp_mutex);
                anonymizer.reset();
                pos_extractor.reset();
                estimator.reset();
                std::cout << "[DETECT]  Tracking lost - downstream modules reset."
                          << std::endl;
            }
            was_tracking = false;
        }

        roi.release();
    }

    roi_queue.shutdown();
    std::cout << "[DETECT]  Thread exiting." << std::endl;
}

// ----------------------------------------------------------------------------
// Thread 3 - Anonymization + DSP + Estimation (Consumer)
// ----------------------------------------------------------------------------

/**
 * @brief Final pipeline stage: anonymise the ROI, extract the BVP signal via
 *        POS, and feed it into the HR / FR estimator.
 *
 * When the estimator completes its calibration phase and produces a 1-second
 * update, the latest vital signs are printed to stdout.
 *
 * @param roi_queue     Incoming 160*208 face ROIs.
 * @param anonymizer    Spatial anonymization module.
 * @param pos_extractor POS BVP extraction module.
 * @param estimator     HR / FR estimation module.
 * @param dsp_mutex     Protects against concurrent reset() from the detection thread.
 */
static void dspThread(BoundedQueue<cv::Mat>& roi_queue,
                      Anonymizer&            anonymizer,
                      PosExtractor&          pos_extractor,
                      estimation&            estimator,
                      std::mutex&            dsp_mutex)
{
    std::cout << "[DSP]    Processing thread started." << std::endl;

    while (g_running.load(std::memory_order_relaxed)) {
        auto maybe_roi = roi_queue.pop(g_running);
        if (!maybe_roi.has_value()) {
            break;  // Shutdown.
        }

        cv::Mat roi = std::move(maybe_roi.value());

        // All three DSP stages share state that the detection thread may
        // reset when tracking is lost, so we hold the mutex throughout.
        std::lock_guard<std::mutex> lock(dsp_mutex);

        // Spatial anonymization (de-identification).
        cv::Mat anonymized = anonymizer.processFrame(roi);
        roi.release();  // Original ROI no longer needed.

        // BVP signal extraction (POS algorithm).
        double bvp = 0.0;
        const bool bvp_ready = pos_extractor.processFrame(anonymized, bvp);
        anonymized.release();

        if (!bvp_ready) {
            continue;  // Still in warm-up phase.
        }

        // HR / FR estimation (IIR filtering + FFT).
        double hr_bpm = 0.0;
        double fr_rpm = 0.0;
        const bool updated = estimator.process_sample(bvp, hr_bpm, fr_rpm);

        if (updated) {
            std::cout << std::fixed << std::setprecision(1)
                      << "[VITALS] HR: " << hr_bpm << " BPM"
                      << " | FR: " << fr_rpm << " RPM"
                      << std::endl;
        }
    }

    std::cout << "[DSP]    Thread exiting." << std::endl;
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------

int main()
{
    // Register signal handlers for graceful shutdown
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    std::cout
        << "============================================================\n"
        << "  PRIVES - Privacy-Preserving Vital Signs Monitor\n"
        << "  Pipeline: Capture -> Detection -> Anonymize -> POS -> Est.\n"
        << "============================================================\n"
        << std::endl;

    // Shared modules (owned by main, referenced by threads)
    Anonymizer   anonymizer;
    PosExtractor pos_extractor;
    estimation   estimator;

    // Mutex that serialises DSP processing against cross-thread resets.
    std::mutex dsp_mutex;

    // Inter-thread queues
    BoundedQueue<cv::Mat> capture_queue(cfg::kCaptureQueueSize);
    BoundedQueue<cv::Mat> roi_queue(cfg::kRoiQueueSize);

    // Launch pipeline threads
    std::thread t_capture(captureThread, std::ref(capture_queue));

    std::thread t_detect(detectionThread,
                         std::ref(capture_queue),
                         std::ref(roi_queue),
                         std::ref(anonymizer),
                         std::ref(pos_extractor),
                         std::ref(estimator),
                         std::ref(dsp_mutex));

    std::thread t_dsp(dspThread,
                      std::ref(roi_queue),
                      std::ref(anonymizer),
                      std::ref(pos_extractor),
                      std::ref(estimator),
                      std::ref(dsp_mutex));

    std::cout << "[MAIN]   All threads launched. Press Ctrl+C to stop.\n"
              << std::endl;

    // Wait for shutdown 
    // The main thread simply waits for the worker threads to finish. The
    // signal handler sets g_running = false, which unblocks the queues and
    // causes all threads to exit their loops.

    if (t_capture.joinable()) t_capture.join();
    capture_queue.shutdown();  // Ensure detection thread wakes up.

    if (t_detect.joinable()) t_detect.join();
    roi_queue.shutdown();      // Ensure DSP thread wakes up.

    if (t_dsp.joinable()) t_dsp.join();

    std::cout << "\n[MAIN]   All threads joined. Shutdown complete." << std::endl;

    return 0;
}
