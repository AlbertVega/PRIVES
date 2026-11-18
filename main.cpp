/**
 * @file main.cpp
 * @brief PRIVES Pipeline Orchestrator.
 *
 * Three-thread architecture:
 *   Thread 1 (Capture)   : GStreamer v4l2src / video file  ->  capture_queue
 *   Thread 2 (Detection) : capture_queue  ->  FaceDetectorSCRFD  ->  roi_queue
 *   Thread 3 (DSP)       : roi_queue  ->  Anonymizer  ->  PosExtractor  ->  estimation
 *
 * Build requirements: OpenCV 4.x, HailoRT, GStreamer 1.x, FFTW3, Iir1, C++17.
 */

#include "detection/FaceDetectorSCRFD.hpp"
#include "anonymizer/Anonymizer.hpp"
#include "extraction/PosExtractor.hpp"
#include "estimation/estimation.hpp"
#include "streaming/MjpegStreamer.hpp"
#include "benchmarks/include/prives_profiler.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <csignal>
#include <optional>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <filesystem>
#include <system_error>

// ----------------------------------------------------------------------------
// Global shutdown & error flags
// ----------------------------------------------------------------------------

static std::atomic<bool> g_running{true};
static std::atomic<bool> g_pipeline_error{false};

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
 * Supports two producer modes:
 *   - push()         : Real-time live camera mode. When full, the oldest
 *                      element is discarded to prevent buffer bloat so the
 *                      consumer always works on the freshest data available.
 *   - pushBlocking() : Offline dataset / video evaluation mode. Blocks the
 *                      producer when full so zero frames are dropped and the
 *                      exact video sampling rate is preserved.
 *
 * Calling close() marks end-of-stream (e.g. video EOF) while allowing
 * consumers to drain all remaining queued items before exiting.
 *
 * @tparam T Element type (must be move-constructible).
 */
template <typename T>
class BoundedQueue
{
public:
    /**
     * @param capacity Maximum number of elements in the queue.
     */
    explicit BoundedQueue(std::size_t capacity) : m_capacity(capacity)
    {
        m_buffer.reserve(capacity);
    }

    /**
     * @brief Push an element into the queue (drop-oldest when full).
     *
     * If the queue is already at capacity the front (oldest) element is
     * discarded so the consumer never processes stale data.
     *
     * @param item Element to enqueue (moved in).
     * @return The evicted element, if any (used for dropped-frame telemetry).
     */
    std::optional<T> push(T item)
    {
        std::optional<T> evicted;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_closed) {
                return std::nullopt;
            }
            if (m_buffer.size() >= m_capacity) {
                // Drop the oldest element (front) to avoid latency build-up.
                evicted = std::move(m_buffer.front());
                m_buffer.erase(m_buffer.begin());
            }
            m_buffer.push_back(std::move(item));
        }
        m_cv.notify_one();
        return evicted;
    }

    /**
     * @brief Push an element into the queue without dropping frames (blocks when full).
     *
     * Used when processing pre-recorded dataset videos (e.g. UBFC-rPPG) so
     * every frame is evaluated sequentially without sample loss.
     *
     * @param item    Element to enqueue (moved in).
     * @param running Reference to the global shutdown flag.
     * @return true if the element was enqueued; false if shutting down or closed.
     */
    bool pushBlocking(T item, const std::atomic<bool>& running)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_buffer.size() >= m_capacity &&
               !m_closed &&
               running.load(std::memory_order_relaxed)) {
            m_cv_not_full.wait_for(lock, std::chrono::milliseconds(100));
        }

        if (m_closed || !running.load(std::memory_order_relaxed)) {
            return false;
        }

        m_buffer.push_back(std::move(item));
        lock.unlock();
        m_cv.notify_one();
        return true;
    }

    /**
     * @brief Pop the front element, blocking until one is available.
     *
     * Returns std::nullopt if the pipeline is shutting down or if the queue
     * has been closed (EOF) and drained empty, so consumer threads exit cleanly.
     *
     * @param running Reference to the global shutdown flag.
     * @return The dequeued element, or std::nullopt on shutdown / drained EOF.
     */
    std::optional<T> pop(const std::atomic<bool>& running)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_buffer.empty() &&
               !m_closed &&
               running.load(std::memory_order_relaxed)) {
            m_cv.wait_for(lock, std::chrono::milliseconds(100));
        }

        if (m_buffer.empty()) {
            return std::nullopt;  // Shutting down or closed & drained.
        }

        T item = std::move(m_buffer.front());
        m_buffer.erase(m_buffer.begin());
        lock.unlock();
        m_cv_not_full.notify_one();
        return item;
    }

    /**
     * @brief Mark end-of-stream so consumers drain remaining items and exit.
     */
    void close()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = true;
        }
        m_cv.notify_all();
        m_cv_not_full.notify_all();
    }

    /**
     * @brief Wake up any thread blocked in pop() or pushBlocking() on shutdown.
     */
    void shutdown()
    {
        close();
    }

private:
    std::size_t              m_capacity;
    std::vector<T>           m_buffer;
    bool                     m_closed = false;
    std::mutex               m_mutex;
    std::condition_variable  m_cv;
    std::condition_variable  m_cv_not_full;
};

// ----------------------------------------------------------------------------
// Frame container (image + per-frame telemetry + metadata)
// ----------------------------------------------------------------------------

namespace bench = prives::bench;

/**
 * @brief Unit of work exchanged between pipeline threads.
 *
 * The timing record is a no-op payload unless PRIVES_PROFILING is defined.
 */
struct PipelineFrame
{
    cv::Mat            image;
    bench::FrameRecord timing;
    std::uint64_t      frame_id      = 0;
    double             video_time_s  = 0.0;
    bool               face_detected = true;
};

// ----------------------------------------------------------------------------
// Pipeline configuration & CLI options
// ----------------------------------------------------------------------------

namespace cfg {
    // GStreamer capture parameters (live camera defaults)
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

    // MJPEG streaming
    constexpr int kStreamPort    = 8080;
    constexpr int kJpegQuality   = 50;

    // Console statistics interval (frames)
    constexpr int kStatsInterval = 30;

    // Default CSV output when running on a video file
    constexpr char kDefaultVideoCsv[] = "prives_estimations.csv";
}

struct RuntimeOptions
{
    std::string video_path;                         // Empty = live camera (/dev/video0)
    std::string csv_path;                           // Empty = disabled (auto-set in --video mode)
    std::string hef_path      = cfg::kHefPath;
    int         stream_port   = cfg::kStreamPort;
    bool        enable_stream = true;
    bool        realtime      = false;              // True = pace video file at 1x native FPS

    bool isVideoMode() const noexcept { return !video_path.empty(); }
    bool hasCsvOutput() const noexcept { return !csv_path.empty(); }
};

static void printUsage(const char* prog_name)
{
    std::cout
        << "Usage: " << prog_name << " [options]\n\n"
        << "Modes:\n"
        << "  (no arguments)            Live camera mode via GStreamer v4l2src (" << cfg::kCameraDevice << ")\n"
        << "  --video <path>            Dataset / video-file mode (e.g. UBFC-rPPG vid.avi)\n\n"
        << "Options:\n"
        << "  --output, --csv <path>    Export per-frame PPG & HR/FR estimations to CSV\n"
        << "                            (defaults to '" << cfg::kDefaultVideoCsv << "' when --video is used)\n"
        << "  --hef <path>              Path to compiled SCRFD .hef file (default: " << cfg::kHefPath << ")\n"
        << "  --realtime                Pace --video playback at 1x native FPS (default: max NPU speed, 0 drops)\n"
        << "  --port <port>             MJPEG HTTP streaming port (default: " << cfg::kStreamPort << ")\n"
        << "  --no-stream               Disable MJPEG HTTP server (recommended for batch dataset runs)\n"
        << "  -h, --help                Show this help message and exit\n"
        << std::endl;
}

static bool parseArgs(int argc, char* argv[], RuntimeOptions& opts, bool& exit_cleanly)
{
    exit_cleanly = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            exit_cleanly = true;
            return false;
        } else if (arg == "--video") {
            if (i + 1 >= argc) {
                std::cerr << "[ARGS] ERROR: --video requires a file path." << std::endl;
                return false;
            }
            opts.video_path = argv[++i];
        } else if (arg == "--output" || arg == "--csv") {
            if (i + 1 >= argc) {
                std::cerr << "[ARGS] ERROR: " << arg << " requires a CSV file path." << std::endl;
                return false;
            }
            opts.csv_path = argv[++i];
        } else if (arg == "--hef") {
            if (i + 1 >= argc) {
                std::cerr << "[ARGS] ERROR: --hef requires a .hef file path." << std::endl;
                return false;
            }
            opts.hef_path = argv[++i];
        } else if (arg == "--port") {
            if (i + 1 >= argc) {
                std::cerr << "[ARGS] ERROR: --port requires a TCP port number." << std::endl;
                return false;
            }
            opts.stream_port = std::atoi(argv[++i]);
        } else if (arg == "--realtime") {
            opts.realtime = true;
        } else if (arg == "--no-stream") {
            opts.enable_stream = false;
        } else {
            std::cerr << "[ARGS] ERROR: Unknown option '" << arg << "'. Use --help for usage." << std::endl;
            return false;
        }
    }

    if (opts.isVideoMode() && opts.csv_path.empty()) {
        opts.csv_path = cfg::kDefaultVideoCsv;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Thread 1 - Capture (Producer)
// ----------------------------------------------------------------------------

/**
 * @brief Continuously grabs frames from either a GStreamer v4l2 live camera
 *        pipeline or a GStreamer/OpenCV video file pipeline and pushes them
 *        into the detection queue.
 *
 * In live camera mode, the GStreamer sink uses `drop=true max-buffers=1` so the
 * kernel driver never stalls if the consumer is slower than the camera.
 * In video file mode (`--video`), frames are read losslessly (`drop=false`) and
 * enqueued with backpressure (`pushBlocking`) so every dataset frame is
 * processed without altering the 30 Hz sampling timeline.
 */
static void captureThread(BoundedQueue<PipelineFrame>& capture_queue,
                          bench::TelemetrySink&        telemetry,
                          const RuntimeOptions&        opts)
{
    cv::VideoCapture cap;
    double video_fps = static_cast<double>(cfg::kCaptureFps);
    long long total_video_frames = 0;
    bool using_gst_file = false;

    if (!opts.isVideoMode()) {
        // Live camera mode via GStreamer v4l2src
        const std::string gst_pipeline =
            "v4l2src device=" + std::string(cfg::kCameraDevice) + " ! "
            "video/x-raw,width="  + std::to_string(cfg::kCaptureWidth)  + ","
            "height=" + std::to_string(cfg::kCaptureHeight) + ","
            "framerate=" + std::to_string(cfg::kCaptureFps) + "/1 ! "
            "videoconvert ! "
            "appsink drop=true max-buffers=1";

        cap.open(gst_pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            std::cerr << "[CAPTURE] ERROR: Failed to open GStreamer pipeline:\n"
                      << "  " << gst_pipeline << "\n"
                      << "  Ensure the camera is connected at " << cfg::kCameraDevice
                      << " and GStreamer v4l2 plugins are installed." << std::endl;
            g_pipeline_error.store(true, std::memory_order_relaxed);
            g_running.store(false, std::memory_order_relaxed);
            capture_queue.shutdown();
            return;
        }

        std::cout << "[CAPTURE] GStreamer camera pipeline opened ("
                  << cfg::kCaptureWidth << "x" << cfg::kCaptureHeight
                  << " @ " << cfg::kCaptureFps << " FPS)." << std::endl;
    } else {
        // Dataset / video-file mode (e.g. UBFC-rPPG)
        if (!std::filesystem::exists(opts.video_path)) {
            std::cerr << "[CAPTURE] ERROR: Video file does not exist: "
                      << opts.video_path << std::endl;
            g_pipeline_error.store(true, std::memory_order_relaxed);
            g_running.store(false, std::memory_order_relaxed);
            capture_queue.shutdown();
            return;
        }

        // Probe container metadata (FPS & frame count) before opening GStreamer appsink
        {
            cv::VideoCapture probe(opts.video_path);
            if (probe.isOpened()) {
                const double probed_fps = probe.get(cv::CAP_PROP_FPS);
                if (std::isfinite(probed_fps) && probed_fps > 1.0) {
                    video_fps = probed_fps;
                }
                const double probed_cnt = probe.get(cv::CAP_PROP_FRAME_COUNT);
                if (std::isfinite(probed_cnt) && probed_cnt > 0.0) {
                    total_video_frames = static_cast<long long>(probed_cnt);
                }
                probe.release();
            }
        }

        // Primary: GStreamer filesrc -> decodebin -> videoconvert -> appsink
        const std::string gst_file_pipeline =
            "filesrc location=\"" + opts.video_path + "\" ! "
            "decodebin ! "
            "videoconvert ! "
            "appsink drop=false max-buffers=4 sync=" +
            std::string(opts.realtime ? "true" : "false");

        if (cap.open(gst_file_pipeline, cv::CAP_GSTREAMER) && cap.isOpened()) {
            using_gst_file = true;
            std::cout << "[CAPTURE] GStreamer video pipeline opened: " << opts.video_path << "\n"
                      << "          Pipeline: " << gst_file_pipeline << std::endl;
        } else {
            std::cout << "[CAPTURE] Note: GStreamer decodebin pipeline could not open '"
                      << opts.video_path << "'. Falling back to OpenCV video backend." << std::endl;
            cap.open(opts.video_path);
        }

        if (!cap.isOpened()) {
            std::cerr << "[CAPTURE] ERROR: Failed to open video file: " << opts.video_path << "\n"
                      << "  Ensure GStreamer plugins (gstreamer1.0-plugins-good / gstreamer1.0-libav) "
                      << "or FFmpeg codecs are installed." << std::endl;
            g_pipeline_error.store(true, std::memory_order_relaxed);
            g_running.store(false, std::memory_order_relaxed);
            capture_queue.shutdown();
            return;
        }

        const double cap_fps = cap.get(cv::CAP_PROP_FPS);
        if (std::isfinite(cap_fps) && cap_fps > 1.0) {
            video_fps = cap_fps;
        }

        std::cout << std::fixed << std::setprecision(2)
                  << "[CAPTURE] Video metadata: " << video_fps << " FPS";
        if (total_video_frames > 0) {
            std::cout << " | ~" << total_video_frames << " frames ("
                      << (static_cast<double>(total_video_frames) / video_fps) << " s)";
        }
        std::cout << (opts.realtime ? " | Mode: Real-time (1x)" : " | Mode: Lossless fast evaluation")
                  << std::endl;

        if (std::fabs(video_fps - estimation::kSampleRate) > 1.5) {
            std::cerr << "[CAPTURE] WARNING: Video FPS (" << video_fps
                      << " Hz) differs noticeably from PRIVES nominal sampling rate ("
                      << estimation::kSampleRate << " Hz)." << std::endl;
        }
    }

    cv::Mat frame;
    std::uint64_t frame_id = 0;
    const bench::OcclusionSchedule occlusion;
    const auto t_playback_origin = std::chrono::steady_clock::now();

    while (g_running.load(std::memory_order_relaxed)) {
        const auto t_read_start = bench::Clock::now();
        if (!cap.read(frame) || frame.empty()) {
            if (!opts.isVideoMode()) {
                // Live camera: transient read failure (USB hiccup, etc.) - retry silently.
                continue;
            }

            // Video file mode: if GStreamer failed on frame 0 (missing demuxer/decoder plugin),
            // attempt transparent fallback to OpenCV's default backend before giving up.
            if (frame_id == 0 && using_gst_file) {
                std::cout << "[CAPTURE] Note: GStreamer failed to decode first frame; "
                          << "retrying with OpenCV default video backend..." << std::endl;
                cap.release();
                using_gst_file = false;
                if (cap.open(opts.video_path) && cap.isOpened() &&
                    cap.read(frame) && !frame.empty()) {
                    // Fallback succeeded; proceed normally with this first frame.
                } else {
                    std::cerr << "[CAPTURE] ERROR: Could not decode any frames from "
                              << opts.video_path << std::endl;
                    g_pipeline_error.store(true, std::memory_order_relaxed);
                    g_running.store(false, std::memory_order_relaxed);
                    capture_queue.shutdown();
                    return;
                }
            } else {
                // Normal End-Of-File (EOF) reached.
                if (frame_id == 0) {
                    std::cerr << "[CAPTURE] ERROR: Video file yielded 0 frames: "
                              << opts.video_path << std::endl;
                    g_pipeline_error.store(true, std::memory_order_relaxed);
                } else {
                    std::cout << "[CAPTURE] End of video reached (" << frame_id
                              << " frames read). Draining pipeline..." << std::endl;
                }
                break;
            }
        }
        const auto t_delivered = bench::Clock::now();

        const std::uint64_t current_id = frame_id++;
        PipelineFrame item;
        item.frame_id     = current_id;
        item.video_time_s = static_cast<double>(current_id) / video_fps;
        item.timing.begin(current_id, t_delivered);
        item.timing.add(bench::Stage::Capture, t_delivered - t_read_start);

        // Push a *copy* so the capture buffer can be reused immediately.
        item.image = frame.clone();

        // Benchmark scenario E2: blank the frame to force tracking loss.
        if (occlusion.occluded(t_delivered)) {
            item.image.setTo(cv::Scalar::all(0));
        }

        if (opts.isVideoMode()) {
            // If using OpenCV fallback in --realtime mode, pace delivery to native FPS.
            if (opts.realtime && !using_gst_file && video_fps > 0.0) {
                const auto target_time = t_playback_origin +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(static_cast<double>(current_id + 1) / video_fps));
                std::this_thread::sleep_until(target_time);
            }

            if (!capture_queue.pushBlocking(std::move(item), g_running)) {
                break;
            }
        } else {
            if (auto evicted = capture_queue.push(std::move(item))) {
                telemetry.commit(evicted->timing, /*dropped=*/true);
            }
        }
    }

    cap.release();
    capture_queue.close();
    std::cout << "[CAPTURE] Thread exiting." << std::endl;
}

// ----------------------------------------------------------------------------
// Thread 2 - Detection (Consumer / Producer)
// ----------------------------------------------------------------------------

/**
 * @brief Runs SCRFD face detection on each captured frame.
 *
 * On detection the 160x208 ROI is forwarded to the DSP queue. Just before
 * releasing the original frame, a composed overlay is pushed to the MJPEG
 * streamer for live browser visualisation (when enabled).
 *
 * When tracking is lost (no detection), all downstream modules are reset so
 * that signal buffers do not mix data from different subjects.
 *
 * @param capture_queue    Incoming frames from the capture thread.
 * @param roi_queue        Outgoing 160x208 ROIs to the DSP thread.
 * @param anonymizer       Shared reference for cross-thread reset.
 * @param pos_extractor    Shared reference for cross-thread reset.
 * @param estimator        Shared reference for cross-thread reset.
 * @param dsp_mutex        Protects reset() calls on DSP-owned objects.
 * @param streamer         Optional MJPEG HTTP streamer for live visualisation.
 * @param shared_hr        Atomic HR value published by the DSP thread.
 * @param shared_fr        Atomic FR value published by the DSP thread.
 * @param vitals_ready     Atomic flag: true once the estimator is calibrated.
 * @param telemetry        Per-frame latency sink (no-op unless profiling is enabled).
 * @param opts             Runtime CLI options.
 */
static void detectionThread(BoundedQueue<PipelineFrame>& capture_queue,
                            BoundedQueue<PipelineFrame>& roi_queue,
                            Anonymizer&                  anonymizer,
                            PosExtractor&                pos_extractor,
                            estimation&                  estimator,
                            std::mutex&                  dsp_mutex,
                            MjpegStreamer*               streamer,
                            std::atomic<double>&         shared_hr,
                            std::atomic<double>&         shared_fr,
                            std::atomic<bool>&           vitals_ready,
                            bench::TelemetrySink&        telemetry,
                            const RuntimeOptions&        opts)
{
    // Initialise the NPU-accelerated detector.
    std::unique_ptr<FaceDetectorSCRFD> detector;
    try {
        detector = std::make_unique<FaceDetectorSCRFD>(opts.hef_path, cfg::kScoreThreshold);
    } catch (const std::exception& e) {
        std::cerr << "[DETECT]  ERROR: " << e.what() << "\n"
                  << "  Tip: Run from the build/ directory containing '" << opts.hef_path
                  << "' or pass --hef <path/to/scrfd_2.5g.hef>." << std::endl;
        g_pipeline_error.store(true, std::memory_order_relaxed);
        g_running.store(false, std::memory_order_relaxed);
        capture_queue.shutdown();
        roi_queue.shutdown();
        return;
    }
    std::cout << "[DETECT]  SCRFD loaded on Hailo-8 NPU (" << opts.hef_path << ")." << std::endl;

    cv::Mat  roi;
    cv::Rect bbox;
    bool     was_tracking = false;

    // FPS / latency tracking.
    int  frame_count = 0;
    auto t_start     = std::chrono::steady_clock::now();

    while (g_running.load(std::memory_order_relaxed)) {
        auto maybe_frame = capture_queue.pop(g_running);
        if (!maybe_frame.has_value()) {
            break;  // Shutdown or drained EOF.
        }

        const auto t_frame = std::chrono::steady_clock::now();

        cv::Mat frame             = std::move(maybe_frame->image);
        bench::FrameRecord timing = maybe_frame->timing;
        const std::uint64_t fid   = maybe_frame->frame_id;
        const double vtime_s      = maybe_frame->video_time_s;

        bool detected = false;
        {
            bench::ScopedStage stage(timing, bench::Stage::Npu);
            detected = detector->processFrame(frame, roi, bbox);
        }

        // Measure detection latency and running FPS.
        const auto t_done = std::chrono::steady_clock::now();
        const double latency_ms =
            std::chrono::duration<double, std::milli>(t_done - t_frame).count();
        ++frame_count;
        const double elapsed_s =
            std::chrono::duration<double>(t_done - t_start).count();
        const double fps = (elapsed_s > 0.0) ? frame_count / elapsed_s : 0.0;

        // Stream the composed frame BEFORE destroying the original.
        if (streamer != nullptr) {
            FrameInfo fi;
            fi.detected     = detected;
            fi.bbox         = bbox;
            fi.roi          = roi;  // Shallow copy - streamer clones internally.
            fi.latency_ms   = latency_ms;
            fi.fps          = fps;
            fi.hr_bpm       = shared_hr.load(std::memory_order_relaxed);
            fi.fr_rpm       = shared_fr.load(std::memory_order_relaxed);
            fi.vitals_ready = vitals_ready.load(std::memory_order_relaxed);

            streamer->pushFrame(frame, fi);
        }

        // Privacy: destroy the original frame immediately.
        frame.release();

        if (detected) {
            was_tracking = true;
            PipelineFrame roi_item{roi.clone(), timing, fid, vtime_s, /*face_detected=*/true};
            if (opts.isVideoMode()) {
                if (!roi_queue.pushBlocking(std::move(roi_item), g_running)) {
                    roi.release();
                    break;
                }
            } else {
                // Forward the ROI (clone to decouple from detector internals).
                if (auto evicted = roi_queue.push(std::move(roi_item))) {
                    telemetry.commit(evicted->timing, /*dropped=*/true);
                }
            }
        } else {
            if (opts.hasCsvOutput()) {
                // Forward an in-band tracking-loss record so dspThread resets state
                // and logs the missing-face frame in exact frame_id order.
                PipelineFrame miss_item{cv::Mat{}, timing, fid, vtime_s, /*face_detected=*/false};
                if (opts.isVideoMode()) {
                    if (!roi_queue.pushBlocking(std::move(miss_item), g_running)) {
                        roi.release();
                        break;
                    }
                } else {
                    if (auto evicted = roi_queue.push(std::move(miss_item))) {
                        telemetry.commit(evicted->timing, /*dropped=*/true);
                    }
                }
                was_tracking = false;
            } else {
                if (was_tracking) {
                    // Tracking was just lost - reset all downstream state.
                    std::lock_guard<std::mutex> lock(dsp_mutex);
                    anonymizer.reset();
                    pos_extractor.reset();
                    estimator.reset();
                    vitals_ready.store(false, std::memory_order_relaxed);
                    std::cout << "[DETECT]  Tracking lost - downstream modules reset."
                              << std::endl;
                }
                was_tracking = false;
                // No face: the frame terminates here.
                telemetry.commit(timing);
            }
        }

        roi.release();

        // Console statistics every N frames.
        if (frame_count % cfg::kStatsInterval == 0) {
            std::cout << std::fixed << std::setprecision(1)
                      << "[STATS]  Frame: " << fid
                      << " | FPS: " << fps
                      << " | Latency: " << latency_ms << " ms"
                      << (detected ? " | Face OK" : " | No face")
                      << std::endl;
        }
    }

    roi_queue.close();
    std::cout << "[DETECT]  Thread exiting (" << frame_count << " frames processed)." << std::endl;
}

// ----------------------------------------------------------------------------
// Thread 3 - Anonymization + DSP + Estimation (Consumer)
// ----------------------------------------------------------------------------

/**
 * @brief Final pipeline stage: anonymise the ROI, extract the BVP signal via
 *        POS, and feed it into the HR / FR estimator.
 *
 * When the estimator produces a 1-second update, the latest vital signs are
 * printed to stdout and published to shared atomics so the detection thread
 * can include them in the MJPEG overlay. When CSV output is enabled, every
 * frame's BVP, bandpass-filtered PPG, and HR/FR estimates are written to disk.
 *
 * @param roi_queue     Incoming 160x208 face ROIs.
 * @param anonymizer    Spatial anonymization module.
 * @param pos_extractor POS BVP extraction module.
 * @param estimator     HR / FR estimation module.
 * @param dsp_mutex     Protects against concurrent reset() from the detection thread.
 * @param shared_hr     Atomic HR output for the streamer overlay.
 * @param shared_fr     Atomic FR output for the streamer overlay.
 * @param vitals_ready  Atomic flag set once estimation is calibrated.
 * @param telemetry     Per-frame latency sink (no-op unless profiling is enabled).
 * @param opts          Runtime CLI options.
 */
static void dspThread(BoundedQueue<PipelineFrame>& roi_queue,
                      Anonymizer&                  anonymizer,
                      PosExtractor&                pos_extractor,
                      estimation&                  estimator,
                      std::mutex&                  dsp_mutex,
                      std::atomic<double>&         shared_hr,
                      std::atomic<double>&         shared_fr,
                      std::atomic<bool>&           vitals_ready,
                      bench::TelemetrySink&        telemetry,
                      const RuntimeOptions&        opts)
{
    std::cout << "[DSP]    Processing thread started." << std::endl;

    std::FILE* csv_file = nullptr;
    std::uint64_t csv_rows_written = 0;
    std::uint64_t hr_updates_count = 0;

    if (opts.hasCsvOutput()) {
        std::error_code ec;
        const std::filesystem::path out_path(opts.csv_path);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path(), ec);
        }
        csv_file = std::fopen(opts.csv_path.c_str(), "w");
        if (csv_file == nullptr) {
            std::cerr << "[CSV]    ERROR: Cannot open output CSV file: "
                      << opts.csv_path << std::endl;
            g_pipeline_error.store(true, std::memory_order_relaxed);
            g_running.store(false, std::memory_order_relaxed);
            return;
        }
        std::setvbuf(csv_file, nullptr, _IOFBF, 1 << 16);
        std::fputs("frame_id,video_time_s,face_detected,bvp_ready,bvp_pos,ppg_filtered,"
                   "hr_ready,vitals_ready,hr_updated,hr_bpm,fr_rpm\n",
                   csv_file);
        std::cout << "[CSV]    Logging estimations -> " << opts.csv_path << std::endl;
    }

    bool   was_tracking_dsp = false;
    bool   has_hr_estimate  = false;
    bool   has_fr_estimate  = false;
    double last_hr_bpm      = 0.0;
    double last_fr_rpm      = 0.0;

    while (g_running.load(std::memory_order_relaxed)) {
        auto maybe_roi = roi_queue.pop(g_running);
        if (!maybe_roi.has_value()) {
            break;  // Shutdown or drained EOF.
        }

        cv::Mat roi               = std::move(maybe_roi->image);
        bench::FrameRecord timing = maybe_roi->timing;
        const std::uint64_t fid   = maybe_roi->frame_id;
        const double vtime_s      = maybe_roi->video_time_s;
        const bool face_ok        = maybe_roi->face_detected;

        // All three DSP stages share state that the detection thread may
        // reset when tracking is lost, so we hold the mutex throughout.
        std::lock_guard<std::mutex> lock(dsp_mutex);

        if (!face_ok) {
            if (was_tracking_dsp) {
                anonymizer.reset();
                pos_extractor.reset();
                estimator.reset();
                vitals_ready.store(false, std::memory_order_relaxed);
                has_hr_estimate = false;
                has_fr_estimate = false;
                last_hr_bpm     = 0.0;
                last_fr_rpm     = 0.0;
                std::cout << "[DETECT]  Tracking lost (frame " << fid
                          << ") - downstream modules reset." << std::endl;
            }
            was_tracking_dsp = false;
            telemetry.commit(timing);

            if (csv_file != nullptr) {
                std::fprintf(csv_file, "%llu,%.6f,0,0,,,0,0,0,,\n",
                             static_cast<unsigned long long>(fid), vtime_s);
                ++csv_rows_written;
            }
            continue;
        }

        was_tracking_dsp = true;

        // Spatial anonymization (de-identification).
        cv::Mat anonymized;
        {
            bench::ScopedStage stage(timing, bench::Stage::Anonymize);
            anonymized = anonymizer.processFrame(roi);
        }
        roi.release();  // Original ROI no longer needed.

        // BVP signal extraction (POS algorithm).
        double bvp = 0.0;
        bool bvp_ready = false;
        {
            bench::ScopedStage stage(timing, bench::Stage::PosDetrend);
            bvp_ready = pos_extractor.processFrame(anonymized, bvp);
        }
        anonymized.release();

        if (!bvp_ready) {
            telemetry.commit(timing);
            if (csv_file != nullptr) {
                std::fprintf(csv_file, "%llu,%.6f,1,0,,,0,0,0,,\n",
                             static_cast<unsigned long long>(fid), vtime_s);
                ++csv_rows_written;
            }
            continue;  // Still in POS warm-up phase (< 60 frames).
        }

        // HR / FR estimation (IIR filtering + FFT).
        double hr_candidate = -1.0;
        double fr_candidate = -1.0;
        bool updated = false;
        {
            // Filter and FFT stages are timed inside the estimation module.
            bench::ActiveRecordScope bind(timing);
            updated = estimator.process_sample(bvp, hr_candidate, fr_candidate);
        }
        telemetry.commit(timing);

        const double ppg_filtered = estimator.lastHrFiltered();
        const bool   hr_ready_now = estimator.isHrReady();
        const bool   all_ready    = estimator.isReady();
        const bool   hr_updated   = (hr_candidate >= 0.0);
        const bool   fr_updated   = (fr_candidate >= 0.0);

        if (hr_updated) {
            last_hr_bpm     = hr_candidate;
            has_hr_estimate = true;
            ++hr_updates_count;
        }
        if (fr_updated) {
            last_fr_rpm     = fr_candidate;
            has_fr_estimate = true;
        }

        if (updated) {
            // Publish to shared atomics for the MJPEG overlay.
            shared_hr.store(last_hr_bpm, std::memory_order_relaxed);
            shared_fr.store(last_fr_rpm, std::memory_order_relaxed);
            vitals_ready.store(true, std::memory_order_relaxed);

            std::cout << std::fixed << std::setprecision(1)
                      << "[VITALS] t=" << vtime_s << " s"
                      << " | HR: " << last_hr_bpm << " BPM"
                      << " | FR: " << last_fr_rpm << " RPM"
                      << std::endl;
        }

        if (csv_file != nullptr) {
            std::fprintf(csv_file, "%llu,%.6f,1,1,%.8f,%.8f,%d,%d,%d,",
                         static_cast<unsigned long long>(fid),
                         vtime_s,
                         bvp,
                         ppg_filtered,
                         hr_ready_now ? 1 : 0,
                         all_ready ? 1 : 0,
                         hr_updated ? 1 : 0);
            if (has_hr_estimate) {
                std::fprintf(csv_file, "%.3f,", last_hr_bpm);
            } else {
                std::fputc(',', csv_file);
            }
            if (has_fr_estimate) {
                std::fprintf(csv_file, "%.3f\n", last_fr_rpm);
            } else {
                std::fputc('\n', csv_file);
            }
            ++csv_rows_written;
        }
    }

    if (csv_file != nullptr) {
        std::fflush(csv_file);
        std::fclose(csv_file);
        csv_file = nullptr;
        std::cout << "[CSV]    Saved " << csv_rows_written << " frame records ("
                  << hr_updates_count << " HR updates) -> " << opts.csv_path
                  << std::endl;
    }

    std::cout << "[DSP]    Thread exiting." << std::endl;
}

// ----------------------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    RuntimeOptions opts;
    bool exit_cleanly = false;
    if (!parseArgs(argc, argv, opts, exit_cleanly)) {
        return exit_cleanly ? 0 : 1;
    }

    // Register signal handlers for graceful shutdown.
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGPIPE, SIG_IGN);  // Prevent crash on broken client socket.

    std::cout
        << "============================================================\n"
        << "  PRIVES - Privacy-Preserving Vital Signs Monitor\n"
        << "  Pipeline: Capture -> Detection -> Anonymize -> POS -> Est.\n"
        << "============================================================\n"
        << std::endl;

    if (opts.isVideoMode()) {
        std::cout << "[MAIN]   Input video : " << opts.video_path << "\n"
                  << "[MAIN]   Output CSV  : " << opts.csv_path << "\n"
                  << "[MAIN]   HEF model   : " << opts.hef_path << "\n"
                  << std::endl;
    }

    // Shared modules (owned by main, referenced by threads).
    Anonymizer   anonymizer;
    PosExtractor pos_extractor;
    estimation   estimator;

    // Mutex that serialises DSP processing against cross-thread resets.
    std::mutex dsp_mutex;

    // Shared vital-signs output (DSP thread writes, detection thread reads for overlay).
    std::atomic<double> shared_hr{0.0};
    std::atomic<double> shared_fr{0.0};
    std::atomic<bool>   vitals_ready{false};

    // Optional MJPEG HTTP streamer (starts its own background accept thread).
    std::unique_ptr<MjpegStreamer> streamer;
    if (opts.enable_stream) {
        try {
            streamer = std::make_unique<MjpegStreamer>(opts.stream_port, cfg::kJpegQuality);
        } catch (const std::exception& e) {
            std::cerr << "[MJPEG]  WARNING: " << e.what()
                      << " (continuing without HTTP stream)." << std::endl;
        }
    }

    // Per-frame latency telemetry (no-op unless built with PRIVES_PROFILING).
    bench::TelemetrySink telemetry;
    telemetry.start();

    // Inter-thread queues.
    BoundedQueue<PipelineFrame> capture_queue(cfg::kCaptureQueueSize);
    BoundedQueue<PipelineFrame> roi_queue(cfg::kRoiQueueSize);

    // Launch pipeline threads.
    std::thread t_capture(captureThread,
                          std::ref(capture_queue),
                          std::ref(telemetry),
                          std::cref(opts));

    std::thread t_detect(detectionThread,
                         std::ref(capture_queue),
                         std::ref(roi_queue),
                         std::ref(anonymizer),
                         std::ref(pos_extractor),
                         std::ref(estimator),
                         std::ref(dsp_mutex),
                         streamer.get(),
                         std::ref(shared_hr),
                         std::ref(shared_fr),
                         std::ref(vitals_ready),
                         std::ref(telemetry),
                         std::cref(opts));

    std::thread t_dsp(dspThread,
                      std::ref(roi_queue),
                      std::ref(anonymizer),
                      std::ref(pos_extractor),
                      std::ref(estimator),
                      std::ref(dsp_mutex),
                      std::ref(shared_hr),
                      std::ref(shared_fr),
                      std::ref(vitals_ready),
                      std::ref(telemetry),
                      std::cref(opts));

    if (opts.isVideoMode()) {
        std::cout << "[MAIN]   Processing video... (Press Ctrl+C to abort early)\n"
                  << std::endl;
    } else {
        std::cout << "[MAIN]   All threads launched. Press Ctrl+C to stop.\n"
                  << std::endl;
    }

    // Wait for shutdown or video completion.
    // In live mode, the signal handler sets g_running = false.
    // In video mode, captureThread closes capture_queue at EOF, which cascades
    // cleanly through detectionThread -> roi_queue -> dspThread.

    if (t_capture.joinable()) t_capture.join();
    capture_queue.close();  // Ensure detection thread wakes up if not already closed.

    if (t_detect.joinable()) t_detect.join();
    roi_queue.close();      // Ensure DSP thread wakes up if not already closed.

    if (t_dsp.joinable()) t_dsp.join();

    // Flush remaining telemetry to disk once all producers have stopped.
    telemetry.stop();

    std::cout << "\n[MAIN]   All threads joined. Shutdown complete." << std::endl;

    return g_pipeline_error.load(std::memory_order_relaxed) ? 1 : 0;
}
