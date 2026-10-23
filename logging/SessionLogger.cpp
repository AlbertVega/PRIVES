/**
 * @file SessionLogger.cpp
 * @brief Implementation of the field-test session recorder (CSV output).
 */

#include "SessionLogger.hpp"

#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

namespace {

/// Local wall-clock time in ISO-8601 with milliseconds, e.g. 2026-10-16T09:15:02.481
std::string isoNow()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()).count() % 1000;

    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S")
       << '.' << std::setw(3) << std::setfill('0') << ms;
    return os.str();
}

/// Format a number for CSV (non-finite -> empty cell).
std::string csvNumber(double v, int precision = 2)
{
    if (!std::isfinite(v)) {
        return "";
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << v;
    return os.str();
}

} // namespace

// ----------------------------------------------------------------------------
// Static utilities
// ----------------------------------------------------------------------------

std::string SessionLogger::sanitizeName(const std::string& raw)
{
    // UTF-8 sequences for common Spanish characters -> ASCII.
    static const std::pair<const char*, char> kTranslit[] = {
        {"\xC3\xA1", 'a'}, {"\xC3\xA9", 'e'}, {"\xC3\xAD", 'i'},
        {"\xC3\xB3", 'o'}, {"\xC3\xBA", 'u'}, {"\xC3\xBC", 'u'},
        {"\xC3\xB1", 'n'},
        {"\xC3\x81", 'A'}, {"\xC3\x89", 'E'}, {"\xC3\x8D", 'I'},
        {"\xC3\x93", 'O'}, {"\xC3\x9A", 'U'}, {"\xC3\x9C", 'U'},
        {"\xC3\x91", 'N'},
    };

    // Trim leading / trailing whitespace.
    const auto first = raw.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = raw.find_last_not_of(" \t\r\n");
    const std::string s = raw.substr(first, last - first + 1);

    std::string out;
    for (std::size_t i = 0; i < s.size();) {
        bool matched = false;
        for (const auto& [seq, ascii] : kTranslit) {
            const std::size_t len = std::char_traits<char>::length(seq);
            if (s.compare(i, len, seq) == 0) {
                out += ascii;
                i += len;
                matched = true;
                break;
            }
        }
        if (matched) {
            continue;
        }

        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isalnum(c) || c == '_' || c == '-') {
            out += static_cast<char>(c);
        } else if (c == ' ' || c == '\t') {
            if (!out.empty() && out.back() != '_') {
                out += '_';
            }
        }
        // Any other byte is dropped.
        ++i;
    }

    // Strip a trailing '_' produced by trailing dropped characters.
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out;
}

std::string SessionLogger::promptSubjectName(std::istream& in, std::ostream& out)
{
    for (;;) {
        out << "Introduzca el nombre: " << std::flush;

        std::string line;
        if (!std::getline(in, line)) {
            out << "\n[LOGGER] No input available - using default name 'sujeto'."
                << std::endl;
            return "sujeto";
        }

        const std::string name = sanitizeName(line);
        if (!name.empty()) {
            return name;
        }
        out << "  Nombre no valido. Use letras, numeros, '_' o '-'." << std::endl;
    }
}

// ----------------------------------------------------------------------------
// Lifecycle
// ----------------------------------------------------------------------------

SessionLogger::~SessionLogger()
{
    close();
}

bool SessionLogger::open(const std::string& subject, const std::string& directory)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_open) {
        return true;
    }

    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec) {
        std::cerr << "[LOGGER] ERROR: cannot create directory '" << directory
                  << "': " << ec.message() << std::endl;
        return false;
    }

    // Find the first free session index for this subject.
    fs::path csv_path;
    for (int index = 1;; ++index) {
        csv_path = fs::path(directory) /
                   (subject + "_" + std::to_string(index) + ".csv");
        if (!fs::exists(csv_path)) {
            break;
        }
    }

    m_csv.open(csv_path, std::ios::out | std::ios::trunc);
    if (!m_csv) {
        std::cerr << "[LOGGER] ERROR: cannot open '" << csv_path.string()
                  << "'." << std::endl;
        return false;
    }

    m_csv_path     = csv_path.string();
    m_t0           = std::chrono::steady_clock::now();
    m_vitals_count = 0;

    m_csv << "type,t_s,timestamp,hr_bpm,fr_rpm,fps,latency_ms,event\n";
    m_csv.flush();

    m_open = true;
    return true;
}

void SessionLogger::close()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_open) {
        return;
    }
    m_csv.close();
    m_open = false;
}

// ----------------------------------------------------------------------------
// Recording
// ----------------------------------------------------------------------------

void SessionLogger::updateDetectionStats(double fps, double latency_ms)
{
    m_fps.store(fps, std::memory_order_relaxed);
    m_latency_ms.store(latency_ms, std::memory_order_relaxed);
}

void SessionLogger::logVitals(double hr_bpm, double fr_rpm)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_open) {
        return;
    }

    m_csv << "vitals,"
          << csvNumber(elapsedSeconds(), 3)                           << ','
          << isoNow()                                                 << ','
          << csvNumber(hr_bpm)                                        << ','
          << csvNumber(fr_rpm)                                        << ','
          << csvNumber(m_fps.load(std::memory_order_relaxed))         << ','
          << csvNumber(m_latency_ms.load(std::memory_order_relaxed))  << ",\n";
    m_csv.flush();

    ++m_vitals_count;
}

void SessionLogger::logEvent(const std::string& event)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_open) {
        return;
    }

    m_csv << "event,"
          << csvNumber(elapsedSeconds(), 3) << ','
          << isoNow()                       << ",,,,,"
          << event                          << '\n';
    m_csv.flush();
}

// ----------------------------------------------------------------------------
// Accessors / internals
// ----------------------------------------------------------------------------

bool SessionLogger::isOpen() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_open;
}

std::string SessionLogger::csvPath() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_csv_path;
}

std::size_t SessionLogger::vitalsCount() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_vitals_count;
}

double SessionLogger::elapsedSeconds() const
{
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - m_t0).count();
}
