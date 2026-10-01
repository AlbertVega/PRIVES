#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../estimation/estimation.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Catch::Matchers::WithinAbs;

static constexpr double kFs = estimation::kSampleRate;  // 30 Hz

// Generates a synthetic BVP sample at time-step n containing an additive
// cardiac + respiratory sinusoidal model.
static double syntheticBVP(int n, double cardiac_hz, double resp_hz)
{
    const double t = static_cast<double>(n) / kFs;
    return std::sin(2.0 * M_PI * cardiac_hz * t) + 0.3 * std::sin(2.0 * M_PI * resp_hz * t);
}

// Runs the estimator for `duration_s` seconds feeding a synthetic signal and
// returns the final HR / FR.
static void runSynthetic(estimation& est, double cardiac_hz, double resp_hz, double duration_s, double& hr_out, double& fr_out)
{
    const int total = static_cast<int>(duration_s * kFs);
    for (int n = 0; n < total; ++n) {
        est.process_sample(syntheticBVP(n, cardiac_hz, resp_hz), hr_out, fr_out);
    }
}

// ---------------------------------------------------------------------------
// Nominal accuracy: 72 BPM / 15 RPM
// ---------------------------------------------------------------------------
TEST_CASE("Estimation converges to 72 BPM and 15 RPM on clean signal", "[estimation][nominal]")
{
    estimation est;
    double hr = 0.0, fr = 0.0;
    runSynthetic(est, 1.2, 0.25, 25.0, hr, fr);

    REQUIRE_THAT(hr, WithinAbs(72.0, 3.0));
    REQUIRE_THAT(fr, WithinAbs(15.0, 2.0));
}

// ---------------------------------------------------------------------------
// Frequency sweep - physiological HR range
// ---------------------------------------------------------------------------
TEST_CASE("Estimation tracks various heart rates", "[estimation][sweep]")
{
    struct Case { double hz; double bpm; double tol; };
    const Case cases[] = {
        {0.75, 45.0,  4.0},   // bradycardia
        {1.17, 70.2,  3.0},   // resting
        {2.00, 120.0, 3.0},   // exercise
        {2.83, 170.0, 4.0},   // near upper bound
    };

    for (const auto& c : cases) {
        SECTION("HR = " + std::to_string(static_cast<int>(c.bpm)) + " BPM")
        {
            estimation est;
            double hr = 0.0, fr = 0.0;
            runSynthetic(est, c.hz, 0.25, 25.0, hr, fr);
            REQUIRE_THAT(hr, WithinAbs(c.bpm, c.tol));
        }
    }
}

// ---------------------------------------------------------------------------
// Frequency sweep - physiological FR range
// ---------------------------------------------------------------------------
TEST_CASE("Estimation tracks various respiratory rates", "[estimation][sweep]")
{
    struct Case { double hz; double rpm; double tol; };
    const Case cases[] = {
        {0.17, 10.2, 2.5},   // slow breathing
        {0.25, 15.0, 2.0},   // normal
        {0.50, 30.0, 2.5},   // fast breathing
    };

    for (const auto& c : cases) {
        SECTION("FR = " + std::to_string(static_cast<int>(c.rpm)) + " RPM")
        {
            estimation est;
            double hr = 0.0, fr = 0.0;
            runSynthetic(est, 1.2, c.hz, 25.0, hr, fr);
            REQUIRE_THAT(fr, WithinAbs(c.rpm, c.tol));
        }
    }
}

// ---------------------------------------------------------------------------
// Out-of-band noise rejection
// ---------------------------------------------------------------------------
TEST_CASE("Estimation rejects out-of-band noise component", "[estimation][rejection]")
{
    estimation est;
    double hr = 0.0, fr = 0.0;

    constexpr double cardiac_hz = 1.2;
    constexpr double resp_hz    = 0.25;
    constexpr double noise_hz   = 5.0;   // well above the cardiac band
    const int total = static_cast<int>(25.0 * kFs);

    for (int n = 0; n < total; ++n) {
        const double t = static_cast<double>(n) / kFs;
        const double bvp = std::sin(2.0 * M_PI * cardiac_hz * t)
                         + 0.3 * std::sin(2.0 * M_PI * resp_hz * t)
                         + 0.5 * std::sin(2.0 * M_PI * noise_hz * t);
        est.process_sample(bvp, hr, fr);
    }

    // The noise should be rejected by the bandpass filters.
    REQUIRE_THAT(hr, WithinAbs(72.0, 3.0));
    REQUIRE_THAT(fr, WithinAbs(15.0, 2.0));
}

// ---------------------------------------------------------------------------
// DC input - should not crash or produce wild values
// ---------------------------------------------------------------------------
TEST_CASE("Estimation handles constant (DC) input without crashing", "[estimation][edge]")
{
    estimation est;
    double hr = 0.0, fr = 0.0;

    const int total = static_cast<int>(15.0 * kFs);
    for (int n = 0; n < total; ++n) {
        est.process_sample(1.0, hr, fr);
    }
    // We don't assert a particular value - just that it's finite and didn't crash.
    REQUIRE(std::isfinite(hr));
    REQUIRE(std::isfinite(fr));
}

// ---------------------------------------------------------------------------
// Calibration timing - isReady and process_sample return value
// ---------------------------------------------------------------------------
TEST_CASE("Estimation isReady becomes true after calibration period", "[estimation][calibration]")
{
    estimation est;
    REQUIRE_FALSE(est.isReady());

    double hr = 0.0, fr = 0.0;

    // Feed samples up to just before FR calibration threshold (300).
    for (int n = 0; n < estimation::kFrCalibSamples - 1; ++n) {
        est.process_sample(syntheticBVP(n, 1.2, 0.25), hr, fr);
    }
    REQUIRE_FALSE(est.isReady());

    // One more sample pushes past the threshold.
    est.process_sample(syntheticBVP(estimation::kFrCalibSamples - 1, 1.2, 0.25), hr, fr);
    REQUIRE(est.isReady());
}

TEST_CASE("process_sample returns true only every 30 samples after calibration", "[estimation][timing]")
{
    estimation est;
    double hr = 0.0, fr = 0.0;

    int update_count = 0;
    const int total = static_cast<int>(20.0 * kFs);  // 600 samples

    for (int n = 0; n < total; ++n) {
        if (est.process_sample(syntheticBVP(n, 1.2, 0.25), hr, fr)) {
            ++update_count;
        }
    }
    // Updates happen every 30 samples. Before calibration threshold there
    // are no updates. After kFrCalibSamples = 300, remaining = 300 samples
    // => 10 update windows.  A few may fire during the gap between HR and FR
    // calibration. Total should be roughly 10.
    REQUIRE(update_count >= 8);
    REQUIRE(update_count <= 12);
}

// ---------------------------------------------------------------------------
// reset() clears all state
// ---------------------------------------------------------------------------
TEST_CASE("Estimation reset clears calibration state", "[estimation][reset]")
{
    estimation est;
    double hr = 0.0, fr = 0.0;

    runSynthetic(est, 1.2, 0.25, 15.0, hr, fr);
    REQUIRE(est.isReady());

    est.reset();
    REQUIRE_FALSE(est.isReady());

    // After reset, the estimator must require the full calibration again.
    for (int n = 0; n < estimation::kFrCalibSamples - 1; ++n) {
        est.process_sample(syntheticBVP(n, 1.2, 0.25), hr, fr);
    }
    REQUIRE_FALSE(est.isReady());
}
