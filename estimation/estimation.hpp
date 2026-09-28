#pragma once

#include <Iir.h>
#include <fftw3.h>
#include <cstddef>

/**
 * @class estimation
 * @brief Real-time Heart Rate (HR) and Respiratory Rate (FR) estimator.
 *
 * Receives the instantaneous BVP signal sample-by-sample and produces:
 *   - HR in beats per minute  (BPM), updated every 1 s after calibration.
 *   - FR in respirations per minute (RPM), updated every 1 s after calibration.
 *
 * Signal conditioning uses IIR Butterworth bandpass filters implemented
 * as cascaded biquad sections via the Iir1 library.
 * Spectral analysis uses FFTW3 real-to-complex transforms with Hamming
 * windowing and zero-padding for sub-BPM frequency resolution.
 */
class estimation {
public:
    // Sampling & timing constants
    static constexpr double kSampleRate     = 30.0;   // Sampling frequency (Hz)
    static constexpr int    kSamplesPerSec  = 30;     // Samples in 1 second

    // HR spectral parameters
    static constexpr int    kHrBufLen       = 300;    // 10 s window @ 30 Hz
    static constexpr int    kHrFftLen       = 2048;   // Zero-padded FFT length
    static constexpr double kHrLowHz        = 0.70;   // Cardiac band lower edge (Hz)
    static constexpr double kHrHighHz       = 3.00;   // Cardiac band upper edge (Hz)

    // FR spectral parameters
    static constexpr int    kFrBufLen       = 450;    // 15 s window @ 30 Hz
    static constexpr int    kFrFftLen       = 2048;   // Zero-padded FFT length
    static constexpr double kFrLowHz        = 0.15;   // Respiratory band lower edge (Hz)
    static constexpr double kFrHighHz       = 0.70;   // Respiratory band upper edge (Hz)

    // Calibration thresholds
    static constexpr int    kHrCalibSamples = 150;    // 5 s warm-up for HR
    static constexpr int    kFrCalibSamples = 300;    // 10 s warm-up for FR

    /**
     * @brief Constructs the estimator: configures IIR filters and allocates FFTW3 plans and memory.
     */
    estimation();

    /**
     * @brief Releases all FFTW3 resources (plans + aligned memory).
     */
    ~estimation();

    // Non-copyable, non-movable (FFTW plans are not copyable).
    estimation(const estimation&)            = delete;
    estimation& operator=(const estimation&) = delete;

    /**
     * @brief Processes one BVP sample through both HR and FR pipelines.
     *
     * Call this method once per frame at 30 Hz. After the calibration phase, 
     * HR and FR outputs are refreshed every 30 samples (1 s).
     *
     * @param bvp_sample  Raw Blood Volume Pulse value from PosExtractor.
     * @param[out] hr_bpm Current heart-rate estimate (BPM). Updated in-place every 1 s.
     * @param[out] fr_rpm Current respiratory-rate estimate (RPM). Updated in-place every 1 s.
     * @return true if both hr_bpm and fr_rpm were recalculated on this call.
     */
    bool process_sample(double bvp_sample, double& hr_bpm, double& fr_rpm);

    /**
     * @brief Resets all internal state (filters, buffers, counters).

     * Call when the face tracker loses the subject or an ROI change invalidates the signal continuity.
     */
    void reset();

    /**
     * @brief Returns true once both pipelines have passed their calibration
     *        phases and are producing valid outputs.
     */
    bool isReady() const;

private:
    // IIR bandpass filters (Butterworth, 4th-order)
    static constexpr int kFilterOrder = 4;

    Iir::Butterworth::BandPass<kFilterOrder> m_hr_filter;
    Iir::Butterworth::BandPass<kFilterOrder> m_fr_filter;

    // Circular buffers (pre-allocated, fixed size)
    double m_hr_buf[kHrBufLen];  // Filtered HR samples
    double m_fr_buf[kFrBufLen];  // Filtered FR samples
    int    m_hr_head;            // Write cursor (HR ring)
    int    m_fr_head;            // Write cursor (FR ring)

    // Hamming window coefficients (pre-computed once)
    double m_hr_window[kHrBufLen];
    double m_fr_window[kFrBufLen];

    // FFTW3 resources
    double*       m_hr_fft_in;    // Real input  [kHrFftLen]
    fftw_complex* m_hr_fft_out;   // Complex output [kHrFftLen/2+1]
    fftw_plan     m_hr_plan;

    double*       m_fr_fft_in;    // Real input  [kFrFftLen]
    fftw_complex* m_fr_fft_out;   // Complex output [kFrFftLen/2+1]
    fftw_plan     m_fr_plan;

    // Counters 
    int m_total_samples;          // Total ingested samples (monotonic)
    int m_update_counter;         // Counts 0..29 then wraps (1 s trigger)

    // Internal helpers
    /**
     * @brief Runs the HR FFT pipeline and returns the dominant frequency in BPM.
     */
    double computeHR() const;

    /**
     * @brief Runs the FR FFT pipeline and returns the dominant frequency in RPM.
     */
    double computeFR() const;

    /**
     * @brief Finds the peak-power frequency bin within [lo_hz, hi_hz] in an
     *        already-executed R2C spectrum and converts to the target unit.
     *
     * @param fft_out  FFTW complex output buffer.
     * @param fft_len  Total FFT length (real side).
     * @param lo_hz    Lower frequency bound (Hz).
     * @param hi_hz    Upper frequency bound (Hz).
     * @param scale    Multiplier to convert Hz to target unit (60.0 for BPM/RPM).
     * @return Estimated value in scaled units (BPM or RPM).
     */
    double peakFrequency(const fftw_complex* fft_out, int fft_len, double lo_hz, double hi_hz, double scale) const;
};
