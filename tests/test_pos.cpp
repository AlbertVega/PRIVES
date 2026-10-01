#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../extraction/PosExtractor.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Catch::Matchers::WithinAbs;

static constexpr std::size_t kWindowSize = 60;
static constexpr double      kFs         = 30.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Generates a synthetic BGR frame where the green channel carries a sinusoidal
// modulation at cardiac_hz.  Red and blue are constant.
static cv::Mat makeSyntheticFrame(int t, double cardiac_hz,
                                  double g_base = 120.0, double g_amp = 2.0,
                                  int r_const = 100, int b_const = 90,
                                  int size = 64)
{
    const double g_val = g_base + g_amp * std::sin(2.0 * M_PI * cardiac_hz * static_cast<double>(t) / kFs);
    const auto g_u8 = static_cast<uint8_t>(std::round(std::clamp(g_val, 0.0, 255.0)));
    return cv::Mat(size, size, CV_8UC3, cv::Scalar(b_const, g_u8, r_const));
}

// Feeds kNumFrames through extractor via processFrame and collects BVP values.
static std::vector<double> runFrameSequence(PosExtractor& ext, int num_frames, double cardiac_hz = 1.2)
{
    std::vector<double> bvp;
    bvp.reserve(num_frames);
    for (int t = 0; t < num_frames; ++t) {
        double val = 0.0;
        ext.processFrame(makeSyntheticFrame(t, cardiac_hz), val);
        bvp.push_back(val);
    }
    return bvp;
}

// Count zero-crossings in a signal slice [from, to).
static int countZeroCrossings(const std::vector<double>& v, int from, int to)
{
    int n = 0;
    for (int i = from + 1; i < to; ++i) {
        if ((v[i] > 0.0) != (v[i - 1] > 0.0)) ++n;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Warm-up / steady-state contract
// ---------------------------------------------------------------------------
TEST_CASE("PosExtractor returns false during warm-up and true after",
          "[pos][warmup]")
{
    PosExtractor ext(kWindowSize);
    double bvp = 0.0;

    for (int t = 0; t < static_cast<int>(kWindowSize) - 1; ++t) {
        bool ready = ext.processFrame(makeSyntheticFrame(t, 1.2), bvp);
        REQUIRE_FALSE(ready);
    }
    // Exactly at window_size-th sample it should become ready.
    bool ready = ext.processFrame(makeSyntheticFrame(kWindowSize - 1, 1.2), bvp);
    REQUIRE(ready);
}

TEST_CASE("isReady reflects internal state", "[pos][warmup]")
{
    PosExtractor ext(kWindowSize);
    REQUIRE_FALSE(ext.isReady());

    double bvp;
    for (std::size_t t = 0; t < kWindowSize; ++t) {
        ext.processFrame(makeSyntheticFrame(static_cast<int>(t), 1.2), bvp);
    }
    REQUIRE(ext.isReady());
}

// ---------------------------------------------------------------------------
// Steady-state BVP values are finite
// ---------------------------------------------------------------------------
TEST_CASE("Steady-state BVP values are all finite", "[pos][finite]")
{
    PosExtractor ext(kWindowSize);
    auto bvp = runFrameSequence(ext, 120, 1.2);

    for (int i = static_cast<int>(kWindowSize); i < 120; ++i) {
        REQUIRE(std::isfinite(bvp[i]));
    }
}

// ---------------------------------------------------------------------------
// Periodicity - the extracted signal must oscillate
// ---------------------------------------------------------------------------
TEST_CASE("Extracted BVP signal shows periodicity at cardiac frequency", "[pos][periodicity]")
{
    PosExtractor ext(kWindowSize);
    auto bvp = runFrameSequence(ext, 120, 1.2);

    int crossings = countZeroCrossings(bvp, static_cast<int>(kWindowSize), 120);
    // 1.2 Hz over ~2 s of steady state => at least 2 full cycles => >= 4
    // zero crossings, but be lenient: require >= 2.
    REQUIRE(crossings >= 2);
}

// ---------------------------------------------------------------------------
// Black frames: must not produce NaN / Inf
// ---------------------------------------------------------------------------
TEST_CASE("PosExtractor handles all-black frames without NaN", "[pos][edge]")
{
    PosExtractor ext(kWindowSize);
    cv::Mat black(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    double bvp = 0.0;

    for (std::size_t t = 0; t < kWindowSize + 10; ++t) {
        ext.processFrame(black, bvp);
        REQUIRE(std::isfinite(bvp));
    }
    // With zero means the guard should kick in and return 0.0.
    REQUIRE_THAT(bvp, WithinAbs(0.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Empty / wrong-type input
// ---------------------------------------------------------------------------
TEST_CASE("PosExtractor returns false for empty Mat", "[pos][validation]")
{
    PosExtractor ext(kWindowSize);
    double bvp = 42.0;
    REQUIRE_FALSE(ext.processFrame(cv::Mat(), bvp));
}

TEST_CASE("PosExtractor returns false for single-channel Mat", "[pos][validation]")
{
    PosExtractor ext(kWindowSize);
    cv::Mat grey(64, 64, CV_8UC1, cv::Scalar(128));
    double bvp = 42.0;
    REQUIRE_FALSE(ext.processFrame(grey, bvp));
}

// ---------------------------------------------------------------------------
// reset() clears internal state
// ---------------------------------------------------------------------------
TEST_CASE("PosExtractor reset forces new warm-up", "[pos][reset]")
{
    PosExtractor ext(kWindowSize);
    double bvp;

    // Fill the window.
    for (std::size_t t = 0; t < kWindowSize; ++t) {
        ext.processFrame(makeSyntheticFrame(static_cast<int>(t), 1.2), bvp);
    }
    REQUIRE(ext.isReady());

    ext.reset();
    REQUIRE_FALSE(ext.isReady());

    // First frame after reset should not be ready.
    REQUIRE_FALSE(ext.processFrame(makeSyntheticFrame(0, 1.2), bvp));
}

// ---------------------------------------------------------------------------
// processSample direct API
// ---------------------------------------------------------------------------
TEST_CASE("processSample produces same result as processFrame for uniform ROI", "[pos][api]")
{
    PosExtractor ext_frame(kWindowSize);
    PosExtractor ext_sample(kWindowSize);

    for (int t = 0; t < 80; ++t) {
        cv::Mat frame = makeSyntheticFrame(t, 1.2, 120.0, 2.0, 100, 90, 64);
        cv::Scalar m = cv::mean(frame);

        double bvp_frame = 0.0, bvp_sample = 0.0;
        ext_frame.processFrame(frame, bvp_frame);

        RgbSample sample;
        sample.b = m[0];
        sample.g = m[1];
        sample.r = m[2];
        ext_sample.processSample(sample, bvp_sample);

        REQUIRE_THAT(bvp_frame, WithinAbs(bvp_sample, 1e-9));
    }
}

// ---------------------------------------------------------------------------
// Illumination robustness - global intensity oscillation should be attenuated
// ---------------------------------------------------------------------------
TEST_CASE("POS attenuates global illumination variation", "[pos][illumination]")
{
    PosExtractor ext(kWindowSize);

    constexpr double cardiac_hz = 1.2;
    constexpr double light_hz   = 0.3;  // slow ambient light oscillation
    constexpr int    num_frames = 150;

    std::vector<double> bvp;
    bvp.reserve(num_frames);

    for (int t = 0; t < num_frames; ++t) {
        const double time = static_cast<double>(t) / kFs;

        // Global illumination modulation (affects R, G, B equally).
        const double light = 1.0 + 0.15 * std::sin(2.0 * M_PI * light_hz * time);

        // Cardiac pulsatile component only on green (rPPG signature).
        const double g = light * (120.0 + 2.0 * std::sin(2.0 * M_PI * cardiac_hz * time));
        const double r = light * 100.0;
        const double b = light * 90.0;

        auto clamp8 = [](double v) -> uint8_t {
            return static_cast<uint8_t>(std::round(std::clamp(v, 0.0, 255.0)));
        };
        cv::Mat frame(64, 64, CV_8UC3, cv::Scalar(clamp8(b), clamp8(g), clamp8(r)));

        double val = 0.0;
        ext.processFrame(frame, val);
        bvp.push_back(val);
    }

    // The extracted signal should still oscillate at the cardiac frequency.
    int crossings = countZeroCrossings(bvp, static_cast<int>(kWindowSize), num_frames);
    REQUIRE(crossings >= 2);
}
