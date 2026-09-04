#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>

/**
 * @brief Stores a single spatially-averaged RGB sample (double precision).
 */
struct RgbSample {
    double r = 0.0;
    double g = 0.0;
    double b = 0.0;
};

/**
 * @class PosExtractor
 * @brief Implements the Plane-Orthogonal-to-Skin (POS) algorithm for
 *        real-time extraction of the Blood Volume Pulse (BVP) signal
 *        from spatially-averaged RGB face ROI frames.
 */
class PosExtractor {
public:
    /**
     * @brief Constructs the extractor and pre-allocates all internal buffers.
     * @param window_size Number of samples in the sliding window (default 60).
     */
    explicit PosExtractor(std::size_t window_size = 60);

    /**
     * @brief Processes a full BGR frame (CV_8UC3) from the Anonymizer module.
     *
     * Performs spatial pooling (channel-wise mean) and delegates to processSample().
     *
     * @param roi_anonymized Anonymized face ROI (CV_8UC3).
     * @param[out] bvp_out   Instantaneous BVP value S(t) when ready.
     * @return true if the window is full and bvp_out is valid; false otherwise.
     */
    bool processFrame(const cv::Mat& roi_anonymized, double& bvp_out);

    /**
     * @brief Processes a pre-computed RGB sample directly.
     *
     * Inserts the sample into the circular buffer and computes the BVP value if the window is full.
     *
     * @param sample An RgbSample with channel means.
     * @param[out] bvp_out Instantaneous BVP value S(t) when ready.
     * @return true if the window is full and bvp_out is valid; false otherwise.
     */
    bool processSample(const RgbSample& sample, double& bvp_out);

    /**
     * @brief Resets the internal state (empties the buffer, resets count).
     *
     * Call this when the face tracker loses the target or the ROI resolution changes abruptly.
     */
    void reset();

    /**
     * @brief Checks whether the sliding window is full (>= L samples ingested).
     * @return true if the window contains L samples and POS output is available.
     */
    bool isReady() const;

private:
    // --- Configuration ---------------------------------------------------
    std::size_t m_window_size;          // L (default 60)

    // --- Circular buffer -------------------------------------------------
    std::vector<RgbSample> m_buffer;    // Pre-allocated ring buffer [L]
    std::size_t m_head;                 // Next write position in ring
    std::size_t m_count;                // Samples ingested so far (capped at L)

    // --- Scratch arrays (pre-allocated, reused every call) ---------------
    std::vector<double> m_rn;           // Normalized R channel [L]
    std::vector<double> m_gn;           // Normalized G channel [L]
    std::vector<double> m_bn;           // Normalized B channel [L]
    std::vector<double> m_x;            // Projection X = Gn - Bn  [L]
    std::vector<double> m_y;            // Projection Y = Gn + Bn - 2*Rn [L]

    /**
     * @brief Runs the POS algorithm over the current window contents and
     *        writes the instantaneous BVP value into bvp_out.
     */
    void computePos(double& bvp_out) const;
};
