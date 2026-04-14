#include "classification.hpp"
#include "image_procesing.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace classify {

std::vector<ShapeDescriptor> loadReferenceDescriptors(
    const std::string& refDir,
    int numShapes)
{
    std::vector<ShapeDescriptor> descriptors;
    descriptors.reserve(numShapes);
    for(int i = 0; i < numShapes; i++){
        std::string fileName = refDir + "/Forma_" + (i < 10? "0": "") + std::to_string(i) + ".png";
        cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
        if(img.empty()){
            std::cerr << "Error: Could not load image " << fileName << std::endl;
            continue;
        }
        cv::Mat binary;
        cv::threshold(img, binary, 128, 255, cv::THRESH_BINARY_INV);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Point> largestContour = imgproc::extractLargestContour(contours);
        if(largestContour.empty()){
            std::cerr << "Error: Could not find contour in image " << fileName << std::endl;
            continue;
        }
        // Resample to fixed size so FFT descriptors always have consistent length
        largestContour = imgproc::resampleContour(largestContour, 256);
        cv::Point2f centroid = imgproc::computeCentroid(largestContour);
        std::array<double, 7> huMoments = imgproc::computeHuMoments(largestContour);
        std::vector<std::complex<double>> signature = imgproc::contourToComplexSignature(largestContour, centroid);
        std::vector<double> fftDescriptors = imgproc::computeFFTDescriptors(signature, 100);
        descriptors.push_back({i, huMoments, fftDescriptors});
    }
    return descriptors;
}

double distanceHuMoments(
    const std::array<double, 7>& a,
    const std::array<double, 7>& b)
{
    double distance = 0.0;
    for(int i = 0; i < 7; i++){
        if(std::abs(a[i]) < 1e-15 || std::abs(b[i]) < 1e-15) continue; 
        distance += std::abs(std::log(std::abs(a[i])) - std::log(std::abs(b[i])));
    }
    return distance;
}

double distanceFFT(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    double distance = 0.0;
    if(a.size() != b.size()){
        std::cerr << "Error: FFT descriptors have different sizes" << std::endl;
        return -1.0;
    }
    for(size_t i = 1; i < a.size(); i++) distance += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(distance);
}

int classifyShape(
    const ShapeDescriptor& query,
    const std::vector<ShapeDescriptor>& refs,
    double huWeight,
    double fftWeight)
{
    double minDistance = std::numeric_limits<double>::max();
    int bestId = -1;
    for(const auto& ref : refs){
        double huDistance = distanceHuMoments(query.huMoments, ref.huMoments);
        double fftDistance = distanceFFT(query.fftDescriptors, ref.fftDescriptors);
        double totalDistance = huWeight * huDistance + fftWeight * fftDistance;
        if(totalDistance < minDistance){
            minDistance = totalDistance;
            bestId = ref.id;
        }
    }
    return bestId;
}

} // namespace classify
