#include "../detection/FaceDetectorSCRFD.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// Signal handling for shutdown
static std::atomic<bool> g_running{true};

static void signalHandler(int /*signum*/)
{
    g_running = false;
}

/**
 * @class MjpegStreamer
 * @brief Serves a live MJPEG stream over HTTP on a given TCP port.
 *
 * The main thread calls updateFrame() with each new composed cv::Mat.
 * A background thread accepts one client at a time, sends HTTP headers,
 * and streams JPEG-encoded frames separated by MJPEG boundary markers.
 */
class MjpegStreamer {
public:
    /**
     * @brief Creates the TCP server socket and starts the background thread.
     * @param port TCP port to listen on (default 8080).
     * @throws std::runtime_error if socket/bind/listen fails.
     */
    explicit MjpegStreamer(int port = 8080) : m_port(port)
    {
        m_server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_server_fd < 0) 
        {
            throw std::runtime_error("MjpegStreamer: socket() failed");
        }

        // Allow quick restart after Ctrl+C
        int opt = 1;
        setsockopt(m_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(static_cast<uint16_t>(port));

        if (bind(m_server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) 
        {
            close(m_server_fd);
            throw std::runtime_error("MjpegStreamer: bind() failed on port " + std::to_string(port));
        }
        if (listen(m_server_fd, 2) < 0) 
        {
            close(m_server_fd);
            throw std::runtime_error("MjpegStreamer: listen() failed");
        }

        m_thread = std::thread(&MjpegStreamer::serverLoop, this);
        std::cout << "[MJPEG] Streaming at http://[IP_ADDRESS]:" << port << std::endl;
    }

    ~MjpegStreamer()
    {
        m_running = false;

        // Unblock accept() by shutting down the server socket
        if (m_server_fd >= 0) 
        {
            shutdown(m_server_fd, SHUT_RDWR);
            close(m_server_fd);
            m_server_fd = -1;
        }
        m_cv.notify_all();

        if (m_thread.joinable()) 
        {
            m_thread.join();
        }
    }

    // Non-copyable
    MjpegStreamer(const MjpegStreamer&)            = delete;
    MjpegStreamer& operator=(const MjpegStreamer&) = delete;

    /**
     * @brief Encodes the frame to JPEG and makes it available for streaming.
     * @param frame The composed BGR frame.
     */
    void updateFrame(const cv::Mat& frame)
    {
        std::vector<uint8_t> jpeg;
        cv::imencode(".jpg", frame, jpeg, {cv::IMWRITE_JPEG_QUALITY, 50});
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_jpeg_buffer = std::move(jpeg);
            m_has_frame = true;
        }
        m_cv.notify_one();
    }

private:
    /**
     * @brief Background thread: accepts clients and streams MJPEG frames.
     *
     * Only one client is served at a time (simple and resource-friendly).
     * When a client disconnects, the loop goes back to accept().
     */
    void serverLoop()
    {
        while (m_running) {
            // Block until a client connects (or shutdown interrupts)
            int client_fd = accept(m_server_fd, nullptr, nullptr);
            if (client_fd < 0) 
            {
                if (!m_running) break;
                continue;   // transient error, retry
            }
            std::cout << "[MJPEG] Client connected." << std::endl;

            // Read and discard the HTTP request line + headers
            char discard[2048];
            recv(client_fd, discard, sizeof(discard), 0);

            // Send the MJPEG HTTP response header
            const std::string header =
                "HTTP/1.0 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                "\r\n";
            if (sendAll(client_fd, header.data(), header.size()) < 0) 
            {
                close(client_fd);
                continue;
            }

            // Stream JPEG frames until client disconnects or shutdown
            while (m_running) 
            {
                std::vector<uint8_t> jpeg;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_cv.wait(lock, [this] { return m_has_frame || !m_running; });
                    if (!m_running) break;
                    jpeg = m_jpeg_buffer;
                    m_has_frame = false;
                }

                // MJPEG part header
                const std::string part =
                    "--frame\r\n"
                    "Content-Type: image/jpeg\r\n"
                    "Content-Length: " + std::to_string(jpeg.size()) + "\r\n"
                    "\r\n";

                if (sendAll(client_fd, part.data(), part.size()) < 0) break;
                if (sendAll(client_fd, reinterpret_cast<const char*>(jpeg.data()), jpeg.size()) < 0) break;
                if (sendAll(client_fd, "\r\n", 2) < 0) break;
            }

            close(client_fd);
            std::cout << "[MJPEG] Client disconnected." << std::endl;
        }
    }

    /**
     * @brief Sends all bytes, retrying on partial writes.
     * @return Total bytes sent, or -1 on error.
     */
    static ssize_t sendAll(int fd, const void* data, size_t len)
    {
        const auto* ptr = static_cast<const char*>(data);
        size_t remaining = len;
        while (remaining > 0) 
        {
            ssize_t sent = send(fd, ptr, remaining, MSG_NOSIGNAL);
            if (sent <= 0) return -1;
            ptr       += sent;
            remaining -= static_cast<size_t>(sent);
        }
        return static_cast<ssize_t>(len);
    }

    int                     m_server_fd = -1;
    int                     m_port;
    std::atomic<bool>       m_running{true};
    std::thread             m_thread;
    std::mutex              m_mutex;
    std::condition_variable m_cv;
    std::vector<uint8_t>    m_jpeg_buffer;
    bool                    m_has_frame = false;
};

// Main
int main()
{
    // Signal handlers
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGPIPE, SIG_IGN);      // Prevent crash on broken socket

    std::cout << "========================================================\n"
              << "  PRIVES - Face Detection Test (SCRFD-500M + Hailo-8)\n"
              << "========================================================\n"
              << std::endl;

    // Open GStreamer capture pipeline
    const std::string gst_pipeline =
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "appsink drop=true max-buffers=1";

    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) 
    {
        std::cerr << "ERROR: Failed to open GStreamer pipeline:\n"
                  << "  " << gst_pipeline << "\n"
                  << "Ensure the USB camera is connected at /dev/video0 and "
                     "GStreamer v4l2 plugins are installed."
                  << std::endl;
        return 1;
    }
    std::cout << "[CAP]  GStreamer pipeline opened (640x480 @ 30 FPS)."
              << std::endl;

    // Initialise face detector (loads HEF onto NPU)
    FaceDetectorSCRFD detector("scrfd_2.5g.hef", 0.5f);
    std::cout << "[DET]  SCRFD-2.5G loaded on Hailo-8 NPU." << std::endl;

    // Start MJPEG HTTP streamer
    constexpr int kStreamPort = 8080;
    MjpegStreamer streamer(kStreamPort);

    // Main processing loop
    cv::Mat  frame;
    cv::Mat  roi;
    cv::Rect bbox;
    int      frame_count = 0;
    auto     t_start     = std::chrono::steady_clock::now();

    std::cout << "\n[RUN]  Entering main loop. Press Ctrl+C to stop.\n" << std::endl;

    while (g_running) 
    {
        const auto t_frame_begin = std::chrono::steady_clock::now();

        // Grab the latest frame (old buffers are dropped by GStreamer)
        if (!cap.read(frame) || frame.empty()) 
        {
            std::cerr << "[WARN] Empty frame, retrying..." << std::endl;
            continue;
        }

        // Run face detection on the NPU
        const bool detected = detector.processFrame(frame, roi, bbox);

        const auto t_frame_end = std::chrono::steady_clock::now();
        const double latency_ms = std::chrono::duration<double, std::milli>(t_frame_end - t_frame_begin).count();

        // Compose the output frame for MJPEG streaming
        if (detected) 
        {
            // Green bounding box on the main frame
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

            // "Face" label above the bbox
            cv::putText(frame, "Face",
                        cv::Point(bbox.x, std::max(bbox.y - 8, 15)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);

            // ROI inset in the top-right corner
            constexpr int kMargin = 10;
            constexpr int kBorder = 2;
            const int inset_x = frame.cols - FaceDetectorSCRFD::kRoiW - kMargin;
            const int inset_y = kMargin;

            // Bounds check (always true for 1280x720, but be safe)
            if (inset_x >= 0 &&
                inset_y >= 0 &&
                inset_x + FaceDetectorSCRFD::kRoiW  <= frame.cols &&
                inset_y + FaceDetectorSCRFD::kRoiH <= frame.rows) {

                // White border around the inset
                cv::rectangle(frame,
                    cv::Rect(inset_x - kBorder, inset_y - kBorder,
                             FaceDetectorSCRFD::kRoiW  + 2 * kBorder,
                             FaceDetectorSCRFD::kRoiH + 2 * kBorder),
                    cv::Scalar(255, 255, 255), cv::FILLED);

                // Copy the 160x208 ROI into the inset
                roi.copyTo(frame(cv::Rect(
                    inset_x, inset_y,
                    FaceDetectorSCRFD::kRoiW,
                    FaceDetectorSCRFD::kRoiH)));

                // Label below the inset
                cv::putText(frame, "ROI 160x208",
                            cv::Point(inset_x, inset_y + FaceDetectorSCRFD::kRoiH + 18),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255), 1);
            }
        } else 
        {
            cv::putText(frame, "No face detected",
                        cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
        }

        // Latency overlay at bottom-left
        {
            const std::string lat_text = "Latency: " + std::to_string(static_cast<int>(latency_ms)) + " ms";
            cv::putText(frame, lat_text,
                        cv::Point(20, frame.rows - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2);
        }

        // Push composed frame to the MJPEG streamer
        streamer.updateFrame(frame);

        // Console statistics (every 30 frames)
        ++frame_count;
        if (frame_count % 30 == 0) 
        {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed_s = std::chrono::duration<double>(now - t_start).count();
            const double avg_fps = frame_count / elapsed_s;

            std::cout << std::fixed << std::setprecision(1)
                      << "[STATS] FPS: " << avg_fps
                      << " | Latency: " << latency_ms << " ms"
                      << (detected ? " | Face OK" : " | No face")
                      << std::endl;
        }

        // Explicitly release frame memory to honour the project's strict
        // policy of destroying the original frame and its copies after
        // each processing cycle (ARM Cortex-A76 memory pressure).
        frame.release();
        roi.release();
    }

    // Shutdown
    cap.release();
    std::cout << "\n[EXIT] Shutting down..." << std::endl;

    return 0;
}
