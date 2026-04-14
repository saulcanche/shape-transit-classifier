#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <array>
#include <complex>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace imgproc {

// Given a vector of contours, returns the one with the largest area.
// Returns an empty vector if contours is empty.
std::vector<cv::Point> extractLargestContour(
    const std::vector<std::vector<cv::Point>>& contours);

// Computes the centroid of a contour using image moments (m10/m00, m01/m00).
cv::Point2f computeCentroid(const std::vector<cv::Point>& contour);

// Returns true if the centroid's x-coordinate is within 'tolerance' pixels
// of the horizontal center of the frame (frameWidth / 2).
bool isAtCenter(const cv::Point2f& centroid, int frameWidth, int tolerance);

// Computes the 7 Hu invariant moments for a contour.
// Uses cv::moments -> cv::HuMoments internally.
std::array<double, 7> computeHuMoments(const std::vector<cv::Point>& contour);

// Converts a contour into a complex 1-D signature.
// For each contour point: r = distance to centroid, theta = angle from centroid.
// Each sample is stored as r * exp(j * theta), i.e. complex(r*cos(theta), r*sin(theta)).
std::vector<std::complex<double>> contourToComplexSignature(
    const std::vector<cv::Point>& contour,
    cv::Point2f centroid);

// Computes FFT descriptors from a complex contour signature.
// Applies DFT, takes magnitudes, normalizes by the DC component,
// and returns the first numDescriptors values.
std::vector<double> computeFFTDescriptors(
    const std::vector<std::complex<double>>& signature,
    int numDescriptors);

// Resamples a contour to exactly numPoints evenly-spaced points along its
// perimeter using linear interpolation.  Ensures a fixed-length signature
// regardless of the original number of contour vertices.
std::vector<cv::Point> resampleContour(
    const std::vector<cv::Point>& contour,
    int numPoints);

} // namespace imgproc

#endif // IMAGE_PROCESSING_HPP
