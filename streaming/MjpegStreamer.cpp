#include "MjpegStreamer.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// ----------------------------------------------------------------------------
// Construction / destruction
// ----------------------------------------------------------------------------

MjpegStreamer::MjpegStreamer(int port, int jpeg_quality)
    : m_port(port), m_jpeg_quality(jpeg_quality)
{
    m_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (m_server_fd < 0) {
        throw std::runtime_error("MjpegStreamer: socket() failed");
    }

    // Allow quick restart after Ctrl+C without TIME_WAIT blocking the port.
    int opt = 1;
    setsockopt(m_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port));

    if (bind(m_server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(m_server_fd);
        throw std::runtime_error(
            "MjpegStreamer: bind() failed on port " + std::to_string(port));
    }
    if (listen(m_server_fd, 2) < 0) {
        close(m_server_fd);
        throw std::runtime_error("MjpegStreamer: listen() failed");
    }

    m_thread = std::thread(&MjpegStreamer::serverLoop, this);
    std::cout << "[MJPEG]  Streaming at http://<IP>:" << port << std::endl;
}

MjpegStreamer::~MjpegStreamer()
{
    m_running = false;

    // Unblock accept() by shutting down the server socket.
    if (m_server_fd >= 0) {
        shutdown(m_server_fd, SHUT_RDWR);
        close(m_server_fd);
        m_server_fd = -1;
    }
    m_cv.notify_all();

    if (m_thread.joinable()) {
        m_thread.join();
    }
}

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

void MjpegStreamer::pushFrame(const cv::Mat& frame, const FrameInfo& info)
{
    // Work on a clone so the caller can release the original immediately.
    cv::Mat canvas = frame.clone();
    composeOverlay(canvas, info);

    // Encode to JPEG and make available for the server thread.
    std::vector<uint8_t> jpeg;
    cv::imencode(".jpg", canvas, jpeg,
                 {cv::IMWRITE_JPEG_QUALITY, m_jpeg_quality});
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_jpeg_buffer = std::move(jpeg);
        m_has_frame = true;
    }
    m_cv.notify_one();
}

// ----------------------------------------------------------------------------
// Overlay composition
// ----------------------------------------------------------------------------

void MjpegStreamer::composeOverlay(cv::Mat& canvas, const FrameInfo& info)
{
    if (info.detected) {
        // Green bounding box
        cv::rectangle(canvas, info.bbox, cv::Scalar(0, 255, 0), 2);

        cv::putText(canvas, "Face",
                    cv::Point(info.bbox.x, std::max(info.bbox.y - 8, 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);

        // ROI inset (top-right corner)
        if (!info.roi.empty()) {
            constexpr int kMargin = 10;
            constexpr int kBorder = 2;
            const int roi_w = info.roi.cols;
            const int roi_h = info.roi.rows;
            const int inset_x = canvas.cols - roi_w - kMargin;
            const int inset_y = kMargin;

            // Bounds check (always true for 640x480, but be defensive).
            if (inset_x >= 0 &&
                inset_y >= 0 &&
                inset_x + roi_w  <= canvas.cols &&
                inset_y + roi_h  <= canvas.rows) {

                // White border around the inset.
                cv::rectangle(canvas,
                    cv::Rect(inset_x - kBorder, inset_y - kBorder, roi_w + 2 * kBorder, roi_h + 2 * kBorder),
                    cv::Scalar(255, 255, 255), cv::FILLED);

                // Copy the ROI into the inset region.
                info.roi.copyTo(canvas(cv::Rect(inset_x, inset_y, roi_w, roi_h)));

                // Label below the inset.
                std::ostringstream roi_label;
                roi_label << "ROI " << roi_w << "x" << roi_h;
                cv::putText(canvas, roi_label.str(),
                            cv::Point(inset_x, inset_y + roi_h + 18),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255), 1);
            }
        }
    } else {
        // No detection indicator
        cv::putText(canvas, "No face detected",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);
    }

    // Semi-transparent HUD background bar (bottom)
    constexpr int kBarHeight = 56;
    const int bar_y = canvas.rows - kBarHeight;
    cv::Mat bar_roi = canvas(cv::Rect(0, bar_y, canvas.cols, kBarHeight));
    bar_roi *= 0.4;  // Darken the bar region for readability.

    // FPS / Latency (bottom-left, line 1)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1)
            << "FPS: " << info.fps
            << "  |  Latency: " << static_cast<int>(info.latency_ms) << " ms";
        cv::putText(canvas, oss.str(),
                    cv::Point(12, canvas.rows - 34),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 255), 1);
    }

    // Vital signs (bottom-left, line 2)
    {
        std::ostringstream oss;
        if (info.vitals_ready) {
            oss << std::fixed << std::setprecision(1)
                << "HR: " << info.hr_bpm << " BPM"
                << "  |  FR: " << info.fr_rpm << " RPM";
        } else {
            oss << "Vitals: calibrating...";
        }
        cv::putText(canvas, oss.str(),
                    cv::Point(12, canvas.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(180, 255, 180), 1);
    }
}

// ----------------------------------------------------------------------------
// Server thread
// ----------------------------------------------------------------------------

void MjpegStreamer::serverLoop()
{
    while (m_running) {
        // Block until a client connects (or shutdown interrupts accept()).
        int client_fd = accept(m_server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (!m_running) break;
            continue;  // Transient error - retry.
        }
        std::cout << "[MJPEG]  Client connected." << std::endl;

        // Read and discard the HTTP request line + headers.
        char discard[2048];
        recv(client_fd, discard, sizeof(discard), 0);

        // Send the MJPEG HTTP response header.
        const std::string header =
            "HTTP/1.0 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "\r\n";
        if (sendAll(client_fd, header.data(), header.size()) < 0) {
            close(client_fd);
            continue;
        }

        // Stream JPEG frames until client disconnects or shutdown.
        while (m_running) {
            std::vector<uint8_t> jpeg;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] { return m_has_frame || !m_running; });
                if (!m_running) break;
                jpeg = m_jpeg_buffer;  // Copy - fast for typical JPEG sizes.
                m_has_frame = false;
            }

            // MJPEG multipart boundary + part header.
            const std::string part =
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: " + std::to_string(jpeg.size()) + "\r\n"
                "\r\n";

            if (sendAll(client_fd, part.data(), part.size()) < 0) break;
            if (sendAll(client_fd, reinterpret_cast<const char*>(jpeg.data()),
                        jpeg.size()) < 0) break;
            if (sendAll(client_fd, "\r\n", 2) < 0) break;
        }

        close(client_fd);
        std::cout << "[MJPEG]  Client disconnected." << std::endl;
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

ssize_t MjpegStreamer::sendAll(int fd, const void* data, size_t len)
{
    const auto* ptr = static_cast<const char*>(data);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t sent = send(fd, ptr, remaining, MSG_NOSIGNAL);
        if (sent <= 0) return -1;
        ptr       += sent;
        remaining -= static_cast<size_t>(sent);
    }
    return static_cast<ssize_t>(len);
}
