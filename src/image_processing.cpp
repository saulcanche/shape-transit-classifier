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
    cv::Moments m = cv::moments(contour, false);
    std::array<double, 7> huMoments;
    cv::HuMoments(m, huMoments);
    return huMoments;
}

std::vector<std::complex<double>> contourToComplexSignature(
    const std::vector<cv::Point>& contour,
    cv::Point2f centroid)
{
    std::vector<std::complex<double>> signature;
    signature.reserve(contour.size());
    for(const auto& p : contour){
        double dx = p.x - centroid.x, dy = p.y - centroid.y;
        double r = std::sqrt(dx*dx + dy*dy), theta = std::atan2(dy, dx);
        signature.emplace_back(r * std::cos(theta), r * std::sin(theta));
    }
    return signature;
}

std::vector<double> computeFFTDescriptors(
    const std::vector<std::complex<double>>& signature,
    int numDescriptors)
{
    std::vector<std::complex<double>> sig_copy(signature);
    cv::Mat mat(1, (int)signature.size(), CV_64FC2, sig_copy.data());
    std::vector<double> descriptors;
    descriptors.reserve(numDescriptors);
    cv::dft(mat, mat, cv::DFT_COMPLEX_OUTPUT);
    for(int i = 0; i < numDescriptors && i < mat.cols; i++){
        double magnitude = std::sqrt(mat.at<cv::Vec2d>(0, i)[0]*mat.at<cv::Vec2d>(0, i)[0] + mat.at<cv::Vec2d>(0, i)[1]*mat.at<cv::Vec2d>(0, i)[1]);
        descriptors.push_back(magnitude);
    }
    return descriptors;
}

} // namespace imgproc
