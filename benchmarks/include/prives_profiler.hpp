/**
 * @file prives_profiler.hpp
 * @brief Low-overhead, per-frame latency instrumentation for the PRIVES pipeline.
 *
 * Design:
 *   - A FrameRecord travels by value together with each frame through the
 *     inter-thread queues. Every stage adds its elapsed time to the record.
 *   - When a frame reaches its terminal stage (or is evicted from a queue) the
 *     record is committed to a TelemetrySink.
 *   - The sink stores records in a pre-allocated, lock-free bounded MPSC ring
 *     buffer. Producers never block nor allocate; if the ring is full the record
 *     is discarded and counted as lost.
 *   - A decoupled writer thread drains the ring periodically and appends rows
 *     to a CSV file, so no disk I/O ever happens on the video threads.
 *
 * Instrumentation is compiled in only when PRIVES_PROFILING is defined.
 * Otherwise every call collapses to a no-op and the writer thread is never
 * started.
 *
 * CSV layout:
 *   frame_id,timestamp_epoch_ms,t_capture_ms,t_npu_ms,t_anon_ms,t_pos_ms,
 *   t_filter_ms,t_fft_ms,t_total_ms,dropped_flag
 *
 * Stages that were not executed for a given frame are written as empty fields.
 * t_total_ms is the wall-clock time from GStreamer frame hand-off to the end
 * of the last executed stage (includes inter-thread queueing). It excludes
 * t_capture_ms, which contains the blocking wait for the next sensor frame.
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <thread>

namespace prives::bench {

#if defined(PRIVES_PROFILING)
inline constexpr bool kProfilingEnabled = true;
#else
inline constexpr bool kProfilingEnabled = false;
#endif

using Clock = std::chrono::steady_clock;

enum class Stage : std::uint8_t {
    Capture = 0,
    Npu,
    Anonymize,
    PosDetrend,
    IirFilter,
    Fft,
    Count
};

inline constexpr std::size_t kStageCount = static_cast<std::size_t>(Stage::Count);

// ----------------------------------------------------------------------------
// FrameRecord
// ----------------------------------------------------------------------------

/**
 * @brief Timing data of a single frame. Trivially copyable (48 bytes on ARM64).
 */
struct FrameRecord {
    static constexpr float kNotExecuted = -1.0f;

    std::uint64_t     frame_id = 0;
    Clock::time_point origin{};            // GStreamer hand-off instant.
    std::array<float, kStageCount> stage_us{
        kNotExecuted, kNotExecuted, kNotExecuted,
        kNotExecuted, kNotExecuted, kNotExecuted};
    float         total_us = kNotExecuted;
    std::uint8_t  dropped  = 0;

    /** @brief Initialises the record when the frame is delivered by the camera. */
    void begin(std::uint64_t id, Clock::time_point t_origin) noexcept
    {
        if constexpr (kProfilingEnabled) {
            frame_id = id;
            origin   = t_origin;
        }
    }

    /** @brief Accumulates a duration into a stage (a stage may run several times per frame). */
    void add(Stage stage, Clock::duration elapsed) noexcept
    {
        if constexpr (kProfilingEnabled) {
            const float us = std::chrono::duration<float, std::micro>(elapsed).count();
            float& slot = stage_us[static_cast<std::size_t>(stage)];
            slot = (slot < 0.0f) ? us : slot + us;
        }
    }
};

// ----------------------------------------------------------------------------
// Scoped timers
// ----------------------------------------------------------------------------

/**
 * @brief RAII timer that adds its lifetime to a stage of an explicit record.
 */
class ScopedStage {
public:
    ScopedStage(FrameRecord& record, Stage stage) noexcept
        : m_record(record), m_stage(stage)
    {
        if constexpr (kProfilingEnabled) {
            m_start = Clock::now();
        }
    }

    ~ScopedStage()
    {
        if constexpr (kProfilingEnabled) {
            m_record.add(m_stage, Clock::now() - m_start);
        }
    }

    ScopedStage(const ScopedStage&)            = delete;
    ScopedStage& operator=(const ScopedStage&) = delete;

private:
    FrameRecord&      m_record;
    Stage             m_stage;
    Clock::time_point m_start{};
};

namespace detail {
inline FrameRecord*& activeRecord() noexcept
{
    thread_local FrameRecord* record = nullptr;
    return record;
}
}  // namespace detail

/**
 * @brief Binds a record to the current thread so that library code (e.g. the
 *        estimation module) can be instrumented without changing its API.
 */
class ActiveRecordScope {
public:
    explicit ActiveRecordScope(FrameRecord& record) noexcept
    {
        if constexpr (kProfilingEnabled) {
            m_previous = detail::activeRecord();
            detail::activeRecord() = &record;
        }
    }

    ~ActiveRecordScope()
    {
        if constexpr (kProfilingEnabled) {
            detail::activeRecord() = m_previous;
        }
    }

    ActiveRecordScope(const ActiveRecordScope&)            = delete;
    ActiveRecordScope& operator=(const ActiveRecordScope&) = delete;

private:
    FrameRecord* m_previous = nullptr;
};

/**
 * @brief RAII timer that adds its lifetime to the thread's active record, if any.
 */
class ScopedActiveStage {
public:
    explicit ScopedActiveStage(Stage stage) noexcept : m_stage(stage)
    {
        if constexpr (kProfilingEnabled) {
            if (detail::activeRecord() != nullptr) {
                m_start = Clock::now();
            }
        }
    }

    ~ScopedActiveStage()
    {
        if constexpr (kProfilingEnabled) {
            if (FrameRecord* record = detail::activeRecord()) {
                record->add(m_stage, Clock::now() - m_start);
            }
        }
    }

    ScopedActiveStage(const ScopedActiveStage&)            = delete;
    ScopedActiveStage& operator=(const ScopedActiveStage&) = delete;

private:
    Stage             m_stage;
    Clock::time_point m_start{};
};

#define PRIVES_PROFILE_CONCAT_IMPL(a, b) a##b
#define PRIVES_PROFILE_CONCAT(a, b)      PRIVES_PROFILE_CONCAT_IMPL(a, b)

/** Times the enclosing scope into the thread's active record (no-op if none is bound). */
#define PRIVES_PROFILE_ACTIVE_STAGE(stage) \
    ::prives::bench::ScopedActiveStage PRIVES_PROFILE_CONCAT(prives_profile_scope_, __LINE__)(stage)

// ----------------------------------------------------------------------------
// Lock-free bounded MPSC ring buffer (Vyukov sequence-based queue)
// ----------------------------------------------------------------------------

template <typename T>
class BoundedRing {
public:
    /** @param capacity_pow2 Capacity, rounded up to the next power of two. */
    explicit BoundedRing(std::size_t capacity_pow2)
    {
        std::size_t cap = 1;
        while (cap < capacity_pow2) {
            cap <<= 1;
        }
        m_mask  = cap - 1;
        m_cells = std::make_unique<Cell[]>(cap);
        for (std::size_t i = 0; i < cap; ++i) {
            m_cells[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    /** @return false if the ring is full (the item is not stored). Wait-free for a single attempt. */
    bool tryPush(const T& item) noexcept
    {
        std::size_t pos = m_enqueue.load(std::memory_order_relaxed);
        for (;;) {
            Cell& cell = m_cells[pos & m_mask];
            const std::size_t seq = cell.sequence.load(std::memory_order_acquire);
            const auto diff = static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(pos);
            if (diff == 0) {
                if (m_enqueue.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    cell.data = item;
                    cell.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = m_enqueue.load(std::memory_order_relaxed);
            }
        }
    }

    /** @brief Single-consumer pop. */
    bool tryPop(T& out) noexcept
    {
        Cell& cell = m_cells[m_dequeue & m_mask];
        const std::size_t seq = cell.sequence.load(std::memory_order_acquire);
        if (static_cast<std::intptr_t>(seq) - static_cast<std::intptr_t>(m_dequeue + 1) < 0) {
            return false;
        }
        out = cell.data;
        cell.sequence.store(m_dequeue + m_mask + 1, std::memory_order_release);
        ++m_dequeue;
        return true;
    }

private:
    struct Cell {
        std::atomic<std::size_t> sequence{0};
        T                        data{};
    };

    std::unique_ptr<Cell[]>             m_cells;
    std::size_t                         m_mask = 0;
    alignas(64) std::atomic<std::size_t> m_enqueue{0};
    alignas(64) std::size_t              m_dequeue = 0;
};

// ----------------------------------------------------------------------------
// TelemetrySink
// ----------------------------------------------------------------------------

/**
 * @brief Collects committed FrameRecords and streams them to CSV from a
 *        background writer thread.
 */
class TelemetrySink {
public:
    static constexpr const char* kDefaultPath = "benchmarks/results/pipeline_telemetry.csv";
    static constexpr const char* kPathEnvVar  = "PRIVES_TELEMETRY_CSV";

    /**
     * @param csv_path      Output file (parent directories are created).
     * @param ring_capacity Number of in-flight records (default ~136 s at 30 FPS).
     * @param flush_period  Writer thread wake-up period.
     */
    explicit TelemetrySink(std::string csv_path = pathFromEnv(),
                           std::size_t ring_capacity = 4096,
                           std::chrono::milliseconds flush_period = std::chrono::milliseconds(250))
        : m_path(std::move(csv_path))
        , m_flush_period(flush_period)
        , m_ring(kProfilingEnabled ? ring_capacity : 1)
    {
        m_anchor_steady = Clock::now();
        m_anchor_epoch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    ~TelemetrySink() { stop(); }

    TelemetrySink(const TelemetrySink&)            = delete;
    TelemetrySink& operator=(const TelemetrySink&) = delete;

    /** @brief Resolves the output path from PRIVES_TELEMETRY_CSV or the default location. */
    static std::string pathFromEnv()
    {
        const char* env = std::getenv(kPathEnvVar);
        return (env != nullptr && *env != '\0') ? std::string(env) : std::string(kDefaultPath);
    }

    /** @brief Opens the CSV file and starts the writer thread. */
    bool start()
    {
        if constexpr (!kProfilingEnabled) {
            return false;
        }
        if (m_running.load(std::memory_order_acquire)) {
            return true;
        }

        std::error_code ec;
        const std::filesystem::path path(m_path);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path(), ec);
        }

        m_file = std::fopen(m_path.c_str(), "w");
        if (m_file == nullptr) {
            std::cerr << "[PROFILER] ERROR: cannot open " << m_path << std::endl;
            return false;
        }
        std::setvbuf(m_file, nullptr, _IOFBF, 1 << 16);
        std::fputs("frame_id,timestamp_epoch_ms,t_capture_ms,t_npu_ms,t_anon_ms,t_pos_ms,"
                   "t_filter_ms,t_fft_ms,t_total_ms,dropped_flag\n", m_file);

        m_running.store(true, std::memory_order_release);
        m_writer = std::thread(&TelemetrySink::writerLoop, this);
        std::cout << "[PROFILER] Telemetry enabled -> " << m_path << std::endl;
        return true;
    }

    /** @brief Stops the writer thread, drains pending records and closes the file. */
    void stop()
    {
        if constexpr (!kProfilingEnabled) {
            return;
        }
        if (!m_running.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        if (m_writer.joinable()) {
            m_writer.join();
        }
        drain();
        std::fclose(m_file);
        m_file = nullptr;

        std::cout << "[PROFILER] " << m_written << " records written, "
                  << m_lost.load(std::memory_order_relaxed)
                  << " lost (ring overflow)." << std::endl;
    }

    /**
     * @brief Finalises and enqueues a frame record. Never blocks.
     * @param record  Frame record (total latency is computed here).
     * @param dropped True if the frame was evicted before completing the pipeline.
     */
    void commit(FrameRecord& record, bool dropped = false) noexcept
    {
        if constexpr (kProfilingEnabled) {
            if (!m_running.load(std::memory_order_relaxed)) {
                return;
            }
            record.dropped = dropped ? 1 : 0;
            record.total_us = dropped
                ? FrameRecord::kNotExecuted
                : std::chrono::duration<float, std::micro>(Clock::now() - record.origin).count();
            if (!m_ring.tryPush(record)) {
                m_lost.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    bool enabled() const noexcept { return kProfilingEnabled && m_running.load(std::memory_order_relaxed); }

private:
    void writerLoop()
    {
        while (m_running.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(m_flush_period);
            drain();
        }
    }

    void drain()
    {
        FrameRecord r;
        bool any = false;
        while (m_ring.tryPop(r)) {
            writeRow(r);
            any = true;
        }
        if (any) {
            std::fflush(m_file);
        }
    }

    void writeRow(const FrameRecord& r)
    {
        const double epoch_ms = m_anchor_epoch_ms +
            std::chrono::duration<double, std::milli>(r.origin - m_anchor_steady).count();

        std::fprintf(m_file, "%llu,%.3f", static_cast<unsigned long long>(r.frame_id), epoch_ms);
        for (const float us : r.stage_us) {
            writeField(us);
        }
        writeField(r.total_us);
        std::fprintf(m_file, ",%u\n", static_cast<unsigned>(r.dropped));
        ++m_written;
    }

    void writeField(float us)
    {
        if (us < 0.0f) {
            std::fputc(',', m_file);
        } else {
            std::fprintf(m_file, ",%.4f", static_cast<double>(us) / 1000.0);
        }
    }

    std::string                m_path;
    std::chrono::milliseconds  m_flush_period;
    BoundedRing<FrameRecord>   m_ring;
    std::atomic<bool>          m_running{false};
    std::atomic<std::uint64_t> m_lost{0};
    std::uint64_t              m_written = 0;
    std::FILE*                 m_file = nullptr;
    std::thread                m_writer;
    Clock::time_point          m_anchor_steady{};
    double                     m_anchor_epoch_ms = 0.0;
};

// ----------------------------------------------------------------------------
// Synthetic face occlusion (scenario E2)
// ----------------------------------------------------------------------------

/**
 * @brief Periodic on/off schedule used to blank frames and force tracking loss.
 *
 * Enabled by PRIVES_BENCH_OCCLUSION_PERIOD_S=<seconds> (profiling builds only).
 * The face is visible for one period, then occluded for the next one, and so on.
 */
class OcclusionSchedule {
public:
    static constexpr const char* kEnvVar = "PRIVES_BENCH_OCCLUSION_PERIOD_S";

    OcclusionSchedule() : m_start(Clock::now())
    {
        if constexpr (kProfilingEnabled) {
            if (const char* env = std::getenv(kEnvVar)) {
                const double seconds = std::atof(env);
                if (seconds > 0.0) {
                    m_period = std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(seconds));
                    std::cout << "[PROFILER] Synthetic occlusion every " << seconds
                              << " s enabled." << std::endl;
                }
            }
        }
    }

    /** @return true if the frame captured at @p now must be blanked. */
    bool occluded(Clock::time_point now) const noexcept
    {
        if constexpr (kProfilingEnabled) {
            if (m_period.count() > 0) {
                return ((now - m_start) / m_period) % 2 == 1;
            }
        }
        return false;
    }

private:
    Clock::time_point m_start;
    Clock::duration   m_period{0};
};

}  // namespace prives::bench
