#pragma once

/**
 * @file SessionLogger.hpp
 * @brief Thread-safe recorder of pipeline readings for field tests.
 *
 * Every session is written to a single CSV file:
 *
 *   <dir>/<subject>_<N>.csv    One row per record (Excel / pandas friendly).
 *
 * N is the first index for which the file does not exist yet, so repeated
 * runs with the same subject never overwrite previous sessions.
 *
 * The file is flushed after every record, so if the process is killed
 * abruptly all readings captured up to that point remain usable.
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iosfwd>
#include <mutex>
#include <string>

class SessionLogger
{
public:
    SessionLogger() = default;
    ~SessionLogger();

    SessionLogger(const SessionLogger&)            = delete;
    SessionLogger& operator=(const SessionLogger&) = delete;

    /**
     * @brief Interactively ask for the subject name until a valid one is given.
     *
     * The name is normalised with sanitizeName(). If the input stream is
     * closed (e.g. no terminal attached) the fallback name "sujeto" is used.
     */
    static std::string promptSubjectName(std::istream& in, std::ostream& out);

    /**
     * @brief Convert free text into a filesystem-safe name.
     *
     * Trims whitespace, replaces inner spaces with '_', transliterates common
     * Spanish accented characters (a-acute -> a, n-tilde -> n, ...) and drops
     * anything that is not [A-Za-z0-9_-].
     */
    static std::string sanitizeName(const std::string& raw);

    /**
     * @brief Create the output directory (if needed) and open the CSV file.
     * @return true on success.
     */
    bool open(const std::string& subject, const std::string& directory = "sessions");

    /**
     * @brief Publish the latest detection-thread statistics.
     *
     * Lock-free; values are attached to the next vitals record.
     */
    void updateDetectionStats(double fps, double latency_ms);

    /** @brief Append a vital-signs record (called on every estimator update). */
    void logVitals(double hr_bpm, double fr_rpm);

    /** @brief Append an event record (e.g. "tracking_lost"). */
    void logEvent(const std::string& event);

    /** @brief Close the CSV file. Idempotent. */
    void close();

    bool        isOpen()      const;
    std::string csvPath()     const;
    std::size_t vitalsCount() const;

private:
    double elapsedSeconds() const;

    mutable std::mutex m_mutex;

    std::ofstream m_csv;
    std::string   m_csv_path;
    bool          m_open = false;

    std::chrono::steady_clock::time_point m_t0{};

    std::size_t m_vitals_count = 0;

    std::atomic<double> m_fps{0.0};
    std::atomic<double> m_latency_ms{0.0};
};
