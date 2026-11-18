#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../anonymizer/Anonymizer.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

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

// ---------------------------------------------------------------------------
// Facial de-identification (ArcFace / SFace via cv::FaceRecognizerSF)
// ---------------------------------------------------------------------------
static std::string resolveAssetPath(const std::string& filename)
{
    namespace fs = std::filesystem;
    const std::string candidates[] = {
        "build/" + filename,
        filename,
        "../build/" + filename,
        "tests/" + filename,
        "../tests/" + filename
    };
    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }
    return "";
}

TEST_CASE("Anonymizer prevents facial re-identification under SFace/ArcFace", "[anonymizer][deidentification]")
{
    constexpr int kNumFaces = 10;
    const std::string model_path = resolveAssetPath("face_recognition_sface_2021dec.onnx");

    if (model_path.empty() || resolveAssetPath("face1.jpg").empty()) {
        SKIP("Skipping SFace de-identification test: face1..10.jpg or face_recognition_sface_2021dec.onnx not found");
    }

    // Load OpenCV SFace (ArcFace-based) recognizer once
    cv::Ptr<cv::FaceRecognizerSF> recognizer = cv::FaceRecognizerSF::create(model_path, "");
    REQUIRE(recognizer != nullptr);

    constexpr double kCosineThreshold = 0.363;
    constexpr double kNormL2Threshold = 1.128;

    double sum_cosine = 0.0;
    double sum_l2     = 0.0;

    std::cout << "\n=== [SFace De-ID] Evaluating " << kNumFaces << " face ROIs (128x176) ===" << std::endl;

    for (int i = 1; i <= kNumFaces; ++i) {
        const std::string filename  = "face" + std::to_string(i) + ".jpg";
        const std::string face_path = resolveAssetPath(filename);
        INFO("Testing image: " << filename);
        REQUIRE_FALSE(face_path.empty());

        const cv::Mat orig_face = cv::imread(face_path, cv::IMREAD_COLOR);
        REQUIRE_FALSE(orig_face.empty());
        REQUIRE(orig_face.type() == CV_8UC3);
        REQUIRE(orig_face.cols == 128);
        REQUIRE(orig_face.rows == 176);

        // Obtain anonymized ROI through Anonymizer (fresh calibration per subject)
        Anonymizer anon(16);
        const cv::Mat anon_face = anon.processFrame(orig_face);
        REQUIRE_FALSE(anon_face.empty());
        REQUIRE(anon_face.size() == orig_face.size());

        // SFace expects 112x112 aligned/cropped face inputs
        cv::Mat orig_aligned, anon_aligned;
        cv::resize(orig_face, orig_aligned, cv::Size(112, 112));
        cv::resize(anon_face, anon_aligned, cv::Size(112, 112));

        cv::Mat e_orig, e_anon;
        recognizer->feature(orig_aligned, e_orig);
        e_orig = e_orig.clone();
        recognizer->feature(anon_aligned, e_anon);
        e_anon = e_anon.clone();

        // Sanity check: comparing original face against itself must yield ~1.0 cosine similarity
        const double self_score = recognizer->match(e_orig, e_orig, cv::FaceRecognizerSF::DisType::FR_COSINE);
        REQUIRE_THAT(self_score, WithinAbs(1.0, 1e-4));

        // Compute de-identification metrics
        const double cosine_score = recognizer->match(e_orig, e_anon, cv::FaceRecognizerSF::DisType::FR_COSINE);
        const double l2_distance  = recognizer->match(e_orig, e_anon, cv::FaceRecognizerSF::DisType::FR_NORM_L2);

        sum_cosine += cosine_score;
        sum_l2     += l2_distance;

        std::cout << "  [" << filename << "] Cosine similarity (orig vs anon): " << cosine_score
                  << " | L2 distance (orig vs anon): " << l2_distance << std::endl;

        INFO("Cosine similarity for " << filename << ": " << cosine_score);
        INFO("L2 distance for " << filename << ": " << l2_distance);

        REQUIRE(cosine_score < kCosineThreshold);
        REQUIRE(l2_distance > kNormL2Threshold);
    }

    const double avg_cosine = sum_cosine / static_cast<double>(kNumFaces);
    const double avg_l2     = sum_l2 / static_cast<double>(kNumFaces);

    std::cout << "-------------------------------------------------------------------\n"
              << "[SFace De-ID Average (" << kNumFaces << " faces)]\n"
              << "  Average Cosine similarity (orig vs anon): " << avg_cosine
              << " (threshold >= " << kCosineThreshold << " for same identity)\n"
              << "  Average L2 distance       (orig vs anon): " << avg_l2
              << " (threshold <= " << kNormL2Threshold << " for same identity)\n"
              << "===================================================================\n" << std::endl;

    REQUIRE(avg_cosine < kCosineThreshold);
    REQUIRE(avg_l2 > kNormL2Threshold);
}
