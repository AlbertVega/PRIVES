#include "../extraction/PosExtractor.hpp"

#include <opencv2/core.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr int    kNumFrames   = 120;
static constexpr int    kFrameSize   = 64;       // 64 × 64 pixels
static constexpr double kFs          = 30.0;     // Sampling rate (FPS)
static constexpr double kHeartRateHz = 1.2;      // 72 BPM
static constexpr double kGreenBase   = 120.0;
static constexpr double kGreenAmp    = 2.0;
static constexpr int    kRedConst    = 100;
static constexpr int    kBlueConst   = 90;
static constexpr std::size_t kWindowSize = 60;

// Helper: generate a synthetic BGR frame for time-step t.
// Green channel carries a sinusoidal modulation; R and B are constant.
static cv::Mat makeSyntheticFrame(int t)
{
    // G(t) = 120.0 + 2.0 * sin(2 * π * 1.2 * t / 30.0)
    const double g_val = kGreenBase + kGreenAmp * std::sin(2.0 * M_PI * kHeartRateHz * static_cast<double>(t) / kFs);

    const auto g_u8 = static_cast<uint8_t>(std::round(std::clamp(g_val, 0.0, 255.0)));

    // OpenCV BGR order -> (Blue, Green, Red)
    cv::Mat frame(kFrameSize, kFrameSize, CV_8UC3, cv::Scalar(kBlueConst, g_u8, kRedConst));
    return frame;
}

int main()
{
    std::printf("=== test_pos: PosExtractor unit test ===\n\n");

    PosExtractor extractor(kWindowSize);

    std::vector<double> bvp_values;
    bvp_values.reserve(kNumFrames);

    int  warm_up_true_count = 0;   // Should stay 0
    int  steady_false_count = 0;   // Should stay 0
    bool all_finite         = true;

    for (int t = 0; t < kNumFrames; ++t) {
        cv::Mat frame = makeSyntheticFrame(t);
        double  bvp   = 0.0;
        bool    ready = extractor.processFrame(frame, bvp);

        if (t < static_cast<int>(kWindowSize) - 1) {
            // Warm-up: must return false
            if (ready) {
                std::printf("[FAIL] Frame %d: expected false during warm-up, got true.\n", t);
                ++warm_up_true_count;
            }
        } else {
            // Steady state: must return true
            if (!ready) {
                std::printf("[FAIL] Frame %d: expected true in steady state, got false.\n", t);
                ++steady_false_count;
            }
            if (!std::isfinite(bvp)) {
                std::printf("[FAIL] Frame %d: BVP is not finite (NaN/Inf).\n", t);
                all_finite = false;
            }
        }

        bvp_values.push_back(bvp);

        std::printf("  t=%3d | ready=%s | bvp=%+.8f\n", t, ready ? "true " : "false", bvp);
    }

    // --- Periodicity check -----------------------------------------------
    // Count sign changes in the steady-state BVP values.  A 1.2 Hz signal
    // over 61 samples at 30 FPS (2.0 s of steady state) should produce
    // at least a couple of zero crossings.
    int sign_changes = 0;
    for (int i = static_cast<int>(kWindowSize); i < kNumFrames; ++i) {
        if (i > static_cast<int>(kWindowSize) &&
            ((bvp_values[i] > 0.0) != (bvp_values[i - 1] > 0.0))) {
            ++sign_changes;
        }
    }

    std::printf("\n--- Summary ---\n");
    std::printf("  Warm-up frames returning true : %d (expected 0)\n", warm_up_true_count);
    std::printf("  Steady frames returning false : %d (expected 0)\n", steady_false_count);
    std::printf("  All steady BVP values finite  : %s\n", all_finite ? "YES" : "NO");
    std::printf("  Sign changes in steady state  : %d (expected >= 2)\n", sign_changes);
    bool pass = (warm_up_true_count == 0) && (steady_false_count == 0) && all_finite && (sign_changes >= 2);

    std::printf("\n=== %s ===\n", pass ? "ALL TESTS PASSED" : "TESTS FAILED");
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
