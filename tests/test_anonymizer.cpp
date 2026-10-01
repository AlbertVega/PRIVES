#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../anonymizer/Anonymizer.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using Catch::Matchers::WithinAbs;

// Generates a synthetic BGR gradient frame whose dimensions are exact multiples
// of macroblock_size. Each pixel gets a unique-ish colour so the permutation
// test has something meaningful to check.
static cv::Mat makeSyntheticFrame(int width, int height)
{
    cv::Mat frame(height, width, CV_8UC3);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            frame.at<cv::Vec3b>(r, c) = cv::Vec3b(
                static_cast<uint8_t>((r * 7 + c * 3) % 256),
                static_cast<uint8_t>((r * 5 + c * 11) % 256),
                static_cast<uint8_t>((r * 3 + c * 7) % 256));
        }
    }
    return frame;
}

// ---------------------------------------------------------------------------
// Mean preservation (spatial invariance)
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer preserves per-channel mean within rounding tolerance", "[anonymizer][invariance]")
{
    constexpr int kMB = 16;
    const cv::Mat frame = makeSyntheticFrame(160, 208);
    Anonymizer anon(kMB);

    const cv::Scalar mean_orig = cv::mean(frame);
    const cv::Mat anon_frame = anon.processFrame(frame);
    const cv::Scalar mean_anon = cv::mean(anon_frame);

    // Integer rounding in YCrCb<->BGR can introduce up to 1.0 per channel.
    for (int c = 0; c < 3; ++c) {
        REQUIRE_THAT(mean_anon[c], WithinAbs(mean_orig[c], 1.5));
    }
}

// ---------------------------------------------------------------------------
// Effective permutation - output must differ from input
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer actually permutes pixel positions", "[anonymizer][permutation]")
{
    const cv::Mat frame = makeSyntheticFrame(160, 208);
    Anonymizer anon(16);

    const cv::Mat anon_frame = anon.processFrame(frame);

    // At least some pixels must have moved.
    REQUIRE(anon_frame.size() == frame.size());
    REQUIRE(anon_frame.type() == frame.type());

    int identical_pixels = 0;
    for (int r = 0; r < frame.rows; ++r) {
        for (int c = 0; c < frame.cols; ++c) {
            if (frame.at<cv::Vec3b>(r, c) == anon_frame.at<cv::Vec3b>(r, c)) {
                ++identical_pixels;
            }
        }
    }
    const int total = frame.rows * frame.cols;
    // With a gradient image, it is statistically impossible for > 50%
    // of pixels to land in the exact same position after permutation.
    REQUIRE(identical_pixels < total / 2);
}

// ---------------------------------------------------------------------------
// Temporal determinism - same sigma yields identical output on the same input
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer is deterministic across consecutive frames", "[anonymizer][determinism]")
{
    const cv::Mat frame = makeSyntheticFrame(160, 208);
    Anonymizer anon(16);

    // t=0 calibrates, t=1 and t=2 reuse the same sigma.
    const cv::Mat t0 = anon.processFrame(frame);
    const cv::Mat t1 = anon.processFrame(frame);
    const cv::Mat t2 = anon.processFrame(frame);

    // t1 and t2 must be byte-identical since the input and sigma are the same.
    cv::Mat diff;
    cv::absdiff(t1, t2, diff);
    REQUIRE(cv::countNonZero(diff.reshape(1)) == 0);
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer rejects empty Mat", "[anonymizer][validation]")
{
    Anonymizer anon(16);
    REQUIRE_THROWS_AS(anon.processFrame(cv::Mat()), std::invalid_argument);
}

TEST_CASE("Anonymizer rejects wrong channel type", "[anonymizer][validation]")
{
    Anonymizer anon(16);
    cv::Mat grey(160, 208, CV_8UC1, cv::Scalar(128));
    REQUIRE_THROWS_AS(anon.processFrame(grey), std::invalid_argument);
}

TEST_CASE("Anonymizer rejects non-multiple dimensions", "[anonymizer][validation]")
{
    Anonymizer anon(16);
    // 161 is not a multiple of 16
    cv::Mat bad(161, 160, CV_8UC3, cv::Scalar(100, 100, 100));
    REQUIRE_THROWS_AS(anon.processFrame(bad), std::invalid_argument);
}

TEST_CASE("Anonymizer rejects macroblock_size <= 0", "[anonymizer][validation]")
{
    REQUIRE_THROWS_AS(Anonymizer(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Anonymizer(-4), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// reset() clears calibration state
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer reset clears calibration", "[anonymizer][reset]")
{
    const cv::Mat frame = makeSyntheticFrame(160, 208);
    Anonymizer anon(16);

    anon.processFrame(frame);
    REQUIRE(anon.isCalibrated());

    anon.reset();
    REQUIRE_FALSE(anon.isCalibrated());

    // After reset, processFrame must recalibrate without crashing.
    const cv::Mat result = anon.processFrame(frame);
    REQUIRE_FALSE(result.empty());
    REQUIRE(anon.isCalibrated());
}

// ---------------------------------------------------------------------------
// Recalibration on resolution change
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer recalibrates on resolution change", "[anonymizer][recalibration]")
{
    Anonymizer anon(16);

    const cv::Mat frame_a = makeSyntheticFrame(160, 208);
    anon.processFrame(frame_a);
    REQUIRE(anon.isCalibrated());

    // Feed a different resolution — should auto-reset and recalibrate.
    const cv::Mat frame_b = makeSyntheticFrame(320, 320);
    const cv::Mat result = anon.processFrame(frame_b);
    REQUIRE_FALSE(result.empty());
    REQUIRE(result.size() == frame_b.size());
}

// ---------------------------------------------------------------------------
// Various macroblock sizes
// ---------------------------------------------------------------------------
TEST_CASE("Anonymizer works with different macroblock sizes", "[anonymizer][macroblock]")
{
    const int mb_size = GENERATE(4, 8, 16, 32);
    // Frame dimensions must be multiples of mb_size.
    const int w = mb_size * 5;
    const int h = mb_size * 4;

    Anonymizer anon(mb_size);
    const cv::Mat frame = makeSyntheticFrame(w, h);
    const cv::Mat result = anon.processFrame(frame);

    REQUIRE(result.size() == frame.size());
    REQUIRE(result.type() == CV_8UC3);

    // Mean preservation check.
    const cv::Scalar m_orig = cv::mean(frame);
    const cv::Scalar m_anon = cv::mean(result);
    for (int c = 0; c < 3; ++c) {
        REQUIRE_THAT(m_anon[c], WithinAbs(m_orig[c], 1.5));
    }
}
