#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../detection/DetectionUtils.hpp"
#include <opencv2/core.hpp>
#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// IoU computation
// ---------------------------------------------------------------------------
TEST_CASE("computeIoU: identical boxes yield 1.0", "[detector][iou]")
{
    cv::Rect a(10, 20, 100, 100);
    REQUIRE_THAT(computeIoU(a, a), WithinAbs(1.0, 1e-5));
}

TEST_CASE("computeIoU: disjoint boxes yield 0.0", "[detector][iou]")
{
    cv::Rect a(0, 0, 50, 50);
    cv::Rect b(200, 200, 50, 50);
    REQUIRE_THAT(computeIoU(a, b), WithinAbs(0.0, 1e-5));
}

TEST_CASE("computeIoU: partial overlap is correct", "[detector][iou]")
{
    cv::Rect a(0, 0, 100, 100);    // area = 10000
    cv::Rect b(50, 0, 100, 100);   // area = 10000, overlap = 50*100 = 5000
    // union = 10000 + 10000 - 5000 = 15000
    const float expected = 5000.0f / 15000.0f;
    REQUIRE_THAT(computeIoU(a, b), WithinAbs(expected, 1e-4));
}

TEST_CASE("computeIoU: containment", "[detector][iou]")
{
    cv::Rect big(0, 0, 200, 200);     // area 40000
    cv::Rect small_r(50, 50, 50, 50); // area 2500
    // intersection = 2500, union = 40000
    REQUIRE_THAT(computeIoU(big, small_r), WithinAbs(2500.0f / 40000.0f, 1e-4));
}

TEST_CASE("computeIoU: zero-area box yields 0.0", "[detector][iou]")
{
    cv::Rect a(0, 0, 0, 0);
    cv::Rect b(0, 0, 100, 100);
    REQUIRE_THAT(computeIoU(a, b), WithinAbs(0.0, 1e-5));
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------
TEST_CASE("NMS: empty input stays empty", "[detector][nms]")
{
    std::vector<Detection> dets;
    nms(dets, 0.4f);
    REQUIRE(dets.empty());
}

TEST_CASE("NMS: single detection is preserved", "[detector][nms]")
{
    std::vector<Detection> dets = {{cv::Rect(10, 10, 50, 50), 0.9f}};
    nms(dets, 0.4f);
    REQUIRE(dets.size() == 1);
    REQUIRE_THAT(dets[0].score, WithinAbs(0.9f, 1e-5));
}

TEST_CASE("NMS: suppresses overlapping lower-score detection", "[detector][nms]")
{
    std::vector<Detection> dets = {
        {cv::Rect(0, 0, 100, 100), 0.95f},
        {cv::Rect(10, 10, 100, 100), 0.80f},  // heavy overlap with [0]
    };
    nms(dets, 0.4f);
    REQUIRE(dets.size() == 1);
    REQUIRE_THAT(dets[0].score, WithinAbs(0.95f, 1e-5));
}

TEST_CASE("NMS: keeps non-overlapping detections", "[detector][nms]")
{
    std::vector<Detection> dets = {
        {cv::Rect(0, 0, 50, 50), 0.9f},
        {cv::Rect(200, 200, 50, 50), 0.85f},
    };
    nms(dets, 0.4f);
    REQUIRE(dets.size() == 2);
}

TEST_CASE("NMS: result is sorted by score descending", "[detector][nms]")
{
    std::vector<Detection> dets = {
        {cv::Rect(0, 0, 50, 50), 0.3f},
        {cv::Rect(200, 200, 50, 50), 0.9f},
        {cv::Rect(400, 400, 50, 50), 0.6f},
    };
    nms(dets, 0.4f);
    REQUIRE(dets.size() == 3);
    REQUIRE(dets[0].score >= dets[1].score);
    REQUIRE(dets[1].score >= dets[2].score);
}

TEST_CASE("NMS: multiple overlapping clusters", "[detector][nms]")
{
    std::vector<Detection> dets = {
        // Cluster A (overlapping)
        {cv::Rect(0, 0, 100, 100), 0.95f},
        {cv::Rect(5, 5, 100, 100), 0.90f},
        {cv::Rect(10, 10, 100, 100), 0.85f},
        // Cluster B (far away, overlapping among themselves)
        {cv::Rect(400, 400, 80, 80), 0.80f},
        {cv::Rect(405, 405, 80, 80), 0.75f},
    };
    nms(dets, 0.4f);
    // One survivor per cluster.
    REQUIRE(dets.size() == 2);
    REQUIRE_THAT(dets[0].score, WithinAbs(0.95f, 1e-5));
    REQUIRE_THAT(dets[1].score, WithinAbs(0.80f, 1e-5));
}

// ---------------------------------------------------------------------------
// Letterbox geometry via computeLetterboxParams
// ---------------------------------------------------------------------------
TEST_CASE("Letterbox: landscape 640x480 input", "[detector][letterbox]")
{
    float scale;
    int pad_left, pad_top;
    computeLetterboxParams(640, 480, 640, scale, pad_left, pad_top);

    REQUIRE_THAT(scale, WithinAbs(1.0f, 1e-5));
    REQUIRE(pad_left == 0);
    REQUIRE(pad_top == 80);
}

TEST_CASE("Letterbox: portrait 480x640 input", "[detector][letterbox]")
{
    float scale;
    int pad_left, pad_top;
    computeLetterboxParams(480, 640, 640, scale, pad_left, pad_top);

    REQUIRE_THAT(scale, WithinAbs(1.0f, 1e-5));
    REQUIRE(pad_left == 80);
    REQUIRE(pad_top == 0);
}

TEST_CASE("Letterbox: widescreen 1920x1080 input", "[detector][letterbox]")
{
    float scale;
    int pad_left, pad_top;
    computeLetterboxParams(1920, 1080, 640, scale, pad_left, pad_top);

    REQUIRE_THAT(scale, WithinAbs(640.0f / 1920.0f, 1e-5));
    REQUIRE(static_cast<int>(1920 * scale) == 640);
    REQUIRE(pad_left == 0);
    REQUIRE(pad_top == 140);
}

TEST_CASE("Letterbox: square input has no padding", "[detector][letterbox]")
{
    float scale;
    int pad_left, pad_top;
    computeLetterboxParams(640, 640, 640, scale, pad_left, pad_top);

    REQUIRE_THAT(scale, WithinAbs(1.0f, 1e-5));
    REQUIRE(pad_left == 0);
    REQUIRE(pad_top == 0);
}

TEST_CASE("Letterbox: small input is upscaled", "[detector][letterbox]")
{
    float scale;
    int pad_left, pad_top;
    computeLetterboxParams(320, 240, 640, scale, pad_left, pad_top);

    REQUIRE_THAT(scale, WithinAbs(2.0f, 1e-5));
    REQUIRE(pad_left == 0);
    REQUIRE(pad_top == 80);
}
