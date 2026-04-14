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
    if(!descriptors.empty() && descriptors[0] != 0.0){
        double dc = descriptors[0];
        for(auto& descriptor: descriptors) descriptor /= dc; 
    }
    return descriptors;
}

std::vector<cv::Point> resampleContour(
    const std::vector<cv::Point>& contour, int numPoints)
{
    if (static_cast<int>(contour.size()) <= 1 || numPoints <= 0)
        return contour;

    // Cumulative arc-length
    std::vector<double> arcLen(contour.size(), 0.0);
    for (size_t i = 1; i < contour.size(); ++i) {
        double dx = contour[i].x - contour[i - 1].x;
        double dy = contour[i].y - contour[i - 1].y;
        arcLen[i] = arcLen[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    double totalLen = arcLen.back();
    if (totalLen < 1e-6) return contour;

    std::vector<cv::Point> resampled(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        double target = totalLen * i / numPoints;
        auto it    = std::lower_bound(arcLen.begin(), arcLen.end(), target);
        size_t idx = static_cast<size_t>(std::distance(arcLen.begin(), it));
        if (idx == 0) {
            resampled[i] = contour[0];
        } else {
            double segLen = arcLen[idx] - arcLen[idx - 1];
            double t = (segLen > 1e-9)
                           ? (target - arcLen[idx - 1]) / segLen
                           : 0.0;
            resampled[i].x = static_cast<int>(
                contour[idx - 1].x + t * (contour[idx].x - contour[idx - 1].x));
            resampled[i].y = static_cast<int>(
                contour[idx - 1].y + t * (contour[idx].y - contour[idx - 1].y));
        }
    }
    return resampled;
}

} // namespace imgproc
