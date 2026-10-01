#include "../estimation/estimation.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main()
{
    estimation est;

    constexpr double fs = estimation::kSampleRate;
    constexpr double cardiac_hz = 1.2;        // 72 BPM
    constexpr double resp_hz    = 0.25;        // 15 RPM
    constexpr double duration_s = 25.0;        // Run for 25 seconds
    const     int    total_samples = static_cast<int>(duration_s * fs);

    double hr_bpm = 0.0;
    double fr_rpm = 0.0;

    int update_count = 0;

    for (int n = 0; n < total_samples; ++n) {
        const double t = static_cast<double>(n) / fs;

        // Synthetic BVP: cardiac sine + additive respiratory baseline wander.
        // Additive model matches real rPPG where respiration causes baseline
        // drift, placing energy directly at 0.25 Hz in the respiratory band.
        const double cardiac = std::sin(2.0 * M_PI * cardiac_hz * t);
        const double resp    = 0.3 * std::sin(2.0 * M_PI * resp_hz * t);
        const double bvp     = cardiac + resp;

        const bool updated = est.process_sample(bvp, hr_bpm, fr_rpm);

        if (updated) {
            ++update_count;
            std::printf("[t=%6.2f s] HR = %6.1f BPM  |  FR = %5.1f RPM\n", t, hr_bpm, fr_rpm);
        }
    }

    std::printf("\n── Final estimates ──\n");
    std::printf("  HR = %.1f BPM  (expected ~72)\n", hr_bpm);
    std::printf("  FR = %.1f RPM  (expected ~15)\n", fr_rpm);
    std::printf("  Updates received: %d\n", update_count);

    // Basic assertions
    const bool hr_ok = std::fabs(hr_bpm - 72.0) < 3.0;
    const bool fr_ok = std::fabs(fr_rpm - 15.0) < 2.0;

    if (hr_ok && fr_ok) {
        std::printf("\n[PASS] All estimates within tolerance.\n");
        return EXIT_SUCCESS;
    } else {
        std::printf("\n[FAIL] Estimates out of tolerance!\n");
        if (!hr_ok) std::printf("  HR error: %.1f BPM\n", std::fabs(hr_bpm - 72.0));
        if (!fr_ok) std::printf("  FR error: %.1f RPM\n", std::fabs(fr_rpm - 15.0));
        return EXIT_FAILURE;
    }
}
