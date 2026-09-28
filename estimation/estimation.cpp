#include "estimation.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

// Construction / Destruction
estimation::estimation()
    : m_hr_head(0)
    , m_fr_head(0)
    , m_hr_fft_in(nullptr)
    , m_hr_fft_out(nullptr)
    , m_fr_fft_in(nullptr)
    , m_fr_fft_out(nullptr)
    , m_total_samples(0)
    , m_update_counter(0)
{
    // Configure IIR bandpass filters
    // Iir1 BandPass::setup(samplingRate, centerFrequency, widthFrequency)
    // centerFreq = (fLow + fHigh) / 2
    // bandwidth  =  fHigh - fLow

    constexpr double hr_center = (kHrLowHz + kHrHighHz) / 2.0;   // 1.85 Hz
    constexpr double hr_width  =  kHrHighHz - kHrLowHz;          // 2.30 Hz
    m_hr_filter.setup(kSampleRate, hr_center, hr_width);

    constexpr double fr_center = (kFrLowHz + kFrHighHz) / 2.0;   // 0.425 Hz
    constexpr double fr_width  =  kFrHighHz - kFrLowHz;          // 0.55 Hz
    m_fr_filter.setup(kSampleRate, fr_center, fr_width);

    // Zero-initialise circular buffers
    std::memset(m_hr_buf, 0, sizeof(m_hr_buf));
    std::memset(m_fr_buf, 0, sizeof(m_fr_buf));

    // Pre-compute Hamming window coefficients
    // w(n) = 0.54 - 0.46 * cos(2*pi*n / (N-1))
    for (int n = 0; n < kHrBufLen; ++n) {
        m_hr_window[n] = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / (kHrBufLen - 1));
    }
    for (int n = 0; n < kFrBufLen; ++n) {
        m_fr_window[n] = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / (kFrBufLen - 1));
    }

    // Allocate FFTW3 resources (aligned memory for ARM NEON)
    m_hr_fft_in  = static_cast<double*>(fftw_malloc(sizeof(double) * kHrFftLen));
    m_hr_fft_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (kHrFftLen / 2 + 1)));
    m_hr_plan = fftw_plan_dft_r2c_1d(kHrFftLen, m_hr_fft_in, m_hr_fft_out, FFTW_MEASURE);

    m_fr_fft_in  = static_cast<double*>(fftw_malloc(sizeof(double) * kFrFftLen));
    m_fr_fft_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (kFrFftLen / 2 + 1)));
    m_fr_plan = fftw_plan_dft_r2c_1d(kFrFftLen, m_fr_fft_in, m_fr_fft_out, FFTW_MEASURE);
}

estimation::~estimation()
{
    if (m_hr_plan)    fftw_destroy_plan(m_hr_plan);
    if (m_fr_plan)    fftw_destroy_plan(m_fr_plan);
    if (m_hr_fft_in)  fftw_free(m_hr_fft_in);
    if (m_hr_fft_out) fftw_free(m_hr_fft_out);
    if (m_fr_fft_in)  fftw_free(m_fr_fft_in);
    if (m_fr_fft_out) fftw_free(m_fr_fft_out);
}

//  Public interface
bool estimation::process_sample(double bvp_sample, double& hr_bpm, double& fr_rpm)
{
    // Apply IIR bandpass filters (sample-by-sample)
    const double hr_filtered = m_hr_filter.filter(bvp_sample);
    const double fr_filtered = m_fr_filter.filter(bvp_sample);

    // Store in respective circular buffers
    m_hr_buf[m_hr_head] = hr_filtered;
    m_hr_head = (m_hr_head + 1) % kHrBufLen;

    m_fr_buf[m_fr_head] = fr_filtered;
    m_fr_head = (m_fr_head + 1) % kFrBufLen;

    ++m_total_samples;
    ++m_update_counter;

    // Every 30 samples (1 s), run spectral analysis
    if (m_update_counter < kSamplesPerSec) {
        return false;
    }
    m_update_counter = 0;

    // Check calibration thresholds before computing
    const bool hr_ready = (m_total_samples >= kHrCalibSamples);
    const bool fr_ready = (m_total_samples >= kFrCalibSamples);

    if (hr_ready) {
        hr_bpm = computeHR();
    }
    if (fr_ready) {
        fr_rpm = computeFR();
    }

    return hr_ready && fr_ready;
}

void estimation::reset()
{
    // Reset filters (Iir1 resets internal delay line)
    constexpr double hr_center = (kHrLowHz + kHrHighHz) / 2.0;
    constexpr double hr_width  =  kHrHighHz - kHrLowHz;
    m_hr_filter.setup(kSampleRate, hr_center, hr_width);

    constexpr double fr_center = (kFrLowHz + kFrHighHz) / 2.0;
    constexpr double fr_width  =  kFrHighHz - kFrLowHz;
    m_fr_filter.setup(kSampleRate, fr_center, fr_width);

    // Clear buffers
    std::memset(m_hr_buf, 0, sizeof(m_hr_buf));
    std::memset(m_fr_buf, 0, sizeof(m_fr_buf));
    m_hr_head = 0;
    m_fr_head = 0;

    // Reset counters
    m_total_samples  = 0;
    m_update_counter = 0;
}

bool estimation::isReady() const
{
    return (m_total_samples >= kFrCalibSamples);
}

// Private: spectral analysis
double estimation::computeHR() const
{
    // Apply Hamming window + zero-padding into the FFT input buffer
    // Read from the circular buffer in chronological order (oldest -> newest).
    // m_hr_head points to the next write position = oldest sample.
    double* in = const_cast<double*>(m_hr_fft_in);

    for (int i = 0; i < kHrBufLen; ++i) {
        const int idx = (m_hr_head + i) % kHrBufLen;
        in[i] = m_hr_buf[idx] * m_hr_window[i];
    }
    // Zero-pad the remainder
    std::memset(in + kHrBufLen, 0, sizeof(double) * (kHrFftLen - kHrBufLen));

    // Execute R2C FFT
    fftw_execute(m_hr_plan);

    // Find dominant peak in [0.70, 3.00] Hz -> BPM
    return peakFrequency(m_hr_fft_out, kHrFftLen, kHrLowHz, kHrHighHz, 60.0);
}

double estimation::computeFR() const
{
    double* in = const_cast<double*>(m_fr_fft_in);

    for (int i = 0; i < kFrBufLen; ++i) {
        const int idx = (m_fr_head + i) % kFrBufLen;
        in[i] = m_fr_buf[idx] * m_fr_window[i];
    }
    std::memset(in + kFrBufLen, 0, sizeof(double) * (kFrFftLen - kFrBufLen));

    fftw_execute(m_fr_plan);

    // Find dominant peak in [0.15, 0.70] Hz -> RPM
    return peakFrequency(m_fr_fft_out, kFrFftLen, kFrLowHz, kFrHighHz, 60.0);
}

double estimation::peakFrequency(const fftw_complex* fft_out, int fft_len, double lo_hz, double hi_hz, double scale) const
{
    // Frequency resolution: df = fs / N_fft
    const double df = kSampleRate / static_cast<double>(fft_len);

    // Bin range of interest (inclusive)
    const int bin_lo = static_cast<int>(std::ceil(lo_hz / df));
    const int bin_hi = std::min(static_cast<int>(std::floor(hi_hz / df)), fft_len / 2);

    double max_power = -1.0;
    int    max_bin   = bin_lo;

    for (int k = bin_lo; k <= bin_hi; ++k) {
        const double re = fft_out[k][0];
        const double im = fft_out[k][1];
        const double power = re * re + im * im;
        if (power > max_power) {
            max_power = power;
            max_bin   = k;
        }
    }

    // Parabolic interpolation for sub-bin accuracy
    // Uses the three bins around the peak to refine the frequency estimate.
    double peak_bin = static_cast<double>(max_bin);

    if (max_bin > bin_lo && max_bin < bin_hi) {
        // Magnitudes of neighbours
        const auto mag = [&](int k) {
            return std::sqrt(fft_out[k][0] * fft_out[k][0] +
                             fft_out[k][1] * fft_out[k][1]);
        };
        const double alpha = mag(max_bin - 1);
        const double beta  = mag(max_bin);
        const double gamma = mag(max_bin + 1);
        const double denom = alpha - 2.0 * beta + gamma;
        if (std::fabs(denom) > 1e-12) {
            peak_bin += 0.5 * (alpha - gamma) / denom;
        }
    }

    const double freq_hz = peak_bin * df;
    return freq_hz * scale;   // Hz -> BPM or RPM
}
