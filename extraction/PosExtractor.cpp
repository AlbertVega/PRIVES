#include "PosExtractor.hpp"

#include <opencv2/core.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

PosExtractor::PosExtractor(std::size_t window_size)
    : m_window_size(window_size)
    , m_head(0)
    , m_count(0)
    , m_buffer(window_size)
    , m_rn(window_size)
    , m_gn(window_size)
    , m_bn(window_size)
    , m_x(window_size)
    , m_y(window_size)
{
}

void PosExtractor::reset()
{
    m_head  = 0;
    m_count = 0;
    // No need to clear buffer contents — they will be overwritten.
}

bool PosExtractor::isReady() const
{
    return m_count >= m_window_size;
}

bool PosExtractor::processFrame(const cv::Mat& roi_anonymized, double& bvp_out)
{
    // Input validation
    if (roi_anonymized.empty()) {
        return false;
    }
    if (roi_anonymized.type() != CV_8UC3) {
        return false;
    }

    // Spatial pooling (channel-wise mean)
    // cv::mean returns (B, G, R, _) for a BGR image.
    cv::Scalar channel_mean = cv::mean(roi_anonymized);

    RgbSample sample;
    sample.b = channel_mean[0];   // OpenCV channel 0 = Blue
    sample.g = channel_mean[1];   // OpenCV channel 1 = Green
    sample.r = channel_mean[2];   // OpenCV channel 2 = Red

    return processSample(sample, bvp_out);
}

bool PosExtractor::processSample(const RgbSample& sample, double& bvp_out)
{
    // Insert into circular buffer
    m_buffer[m_head] = sample;
    m_head = (m_head + 1) % m_window_size;

    if (m_count < m_window_size) {
        ++m_count;
    }

    // Warm-up phase: not enough data yet
    if (m_count < m_window_size) {
        bvp_out = 0.0;
        return false;
    }

    // Steady state: compute POS
    computePos(bvp_out);
    return true;
}

void PosExtractor::computePos(double& bvp_out) const
{
    const std::size_t L = m_window_size;

    // Compute temporal means over the window
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;

    // Iterate the ring buffer starting from the oldest sample.
    // Oldest sample is at index m_head (it was just overwritten on the
    // previous cycle, so after a full window it is the oldest).
    for (std::size_t i = 0; i < L; ++i) {
        std::size_t idx = (m_head + i) % L;   // oldest -> newest
        sum_r += m_buffer[idx].r;
        sum_g += m_buffer[idx].g;
        sum_b += m_buffer[idx].b;
    }

    const double mu_r = sum_r / static_cast<double>(L);
    const double mu_g = sum_g / static_cast<double>(L);
    const double mu_b = sum_b / static_cast<double>(L);

    // Guard: if any mean is essentially zero, we can't normalise.
    constexpr double kEpsilon = 1e-10;
    if (std::fabs(mu_r) < kEpsilon ||
        std::fabs(mu_g) < kEpsilon ||
        std::fabs(mu_b) < kEpsilon) {
        bvp_out = 0.0;
        return;
    }

    // Temporal normalisation and orthogonal projections X, Y
    auto& rn = const_cast<std::vector<double>&>(m_rn);
    auto& gn = const_cast<std::vector<double>&>(m_gn);
    auto& bn = const_cast<std::vector<double>&>(m_bn);
    auto& x  = const_cast<std::vector<double>&>(m_x);
    auto& y  = const_cast<std::vector<double>&>(m_y);

    const double inv_mu_r = 1.0 / mu_r;
    const double inv_mu_g = 1.0 / mu_g;
    const double inv_mu_b = 1.0 / mu_b;

    for (std::size_t i = 0; i < L; ++i) {
        std::size_t idx = (m_head + i) % L;
        rn[i] = m_buffer[idx].r * inv_mu_r;
        gn[i] = m_buffer[idx].g * inv_mu_g;
        bn[i] = m_buffer[idx].b * inv_mu_b;

        x[i] = gn[i] - bn[i];
        y[i] = gn[i] + bn[i] - 2.0 * rn[i];
    }

    // Population standard deviations
    double sum_x = 0.0, sum_y = 0.0;
    for (std::size_t i = 0; i < L; ++i) {
        sum_x += x[i];
        sum_y += y[i];
    }
    const double mu_x = sum_x / static_cast<double>(L);
    const double mu_y = sum_y / static_cast<double>(L);

    double var_x = 0.0, var_y = 0.0;
    for (std::size_t i = 0; i < L; ++i) {
        const double dx = x[i] - mu_x;
        const double dy = y[i] - mu_y;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    var_x /= static_cast<double>(L);
    var_y /= static_cast<double>(L);

    const double sigma_x = std::sqrt(var_x);
    const double sigma_y = std::sqrt(var_y);

    // Adaptive fusion
    constexpr double kSigmaGuard = 1e-6;
    const double alpha = (sigma_y < kSigmaGuard) ? 0.0 : sigma_x / sigma_y;

    // We only need S(L-1), but computing the last element directly:
    bvp_out = x[L - 1] + alpha * y[L - 1];
}
