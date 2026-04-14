#include "image_procesing.hpp"

namespace imgproc {

std::vector<cv::Point> extractLargestContour(
    const std::vector<std::vector<cv::Point>>& contours)
{
    if(contours.empty()) return {};
    double maxArea = 0;
    size_t idx = 0;
    for(size_t i = 0; i < contours.size(); i++){
        double area = cv::contourArea(contours[i]);
        if(area > maxArea) maxArea = area, idx = i;
    }
    return contours[idx];
}

cv::Point2f computeCentroid(const std::vector<cv::Point>& contour)
{
    // TODO: implement — use cv::moments, return (m10/m00, m01/m00)
    cv::Moments m = cv::moments(contour, false);
    if (m.m00 == 0) return {0.0f, 0.0f};
    return {static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00)};
}

bool isAtCenter(const cv::Point2f& centroid, int frameWidth, int tolerance)
{
    return std::abs(centroid.x - frameWidth/2) < tolerance;
}

std::array<double, 7> computeHuMoments(const std::vector<cv::Point>& contour)
{
    // TODO: implement — cv::moments -> cv::HuMoments -> copy to array
    (void)contour;
    return {};
}

std::vector<std::complex<double>> contourToComplexSignature(
    const std::vector<cv::Point>& contour,
    cv::Point2f centroid)
{
    // TODO: implement — for each point: r, theta -> complex(r*cos(theta), r*sin(theta))
    (void)contour;
    (void)centroid;
    return {};
}

std::vector<double> computeFFTDescriptors(
    const std::vector<std::complex<double>>& signature,
    int numDescriptors)
{
    // TODO: implement — cv::dft, magnitudes, normalize by DC, return first N
    (void)signature;
    (void)numDescriptors;
    return {};
}

} // namespace imgproc
