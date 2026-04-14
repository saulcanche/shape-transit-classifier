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
    // TODO: implement — for each Forma_XX.png: load, grayscale, threshold,
    //   findContours, extractLargestContour, computeHuMoments, computeFFTDescriptors
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
        cv::threshold(img, binary, 128, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Point> largestContour = imgproc::extractLargestContour(contours);
        if(largestContour.empty()){
            std::cerr << "Error: Could not find contour in image " << fileName << std::endl;
            continue;
        }
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
    // TODO: implement — sum of |log|a_i| - log|b_i|| for i = 0..6
    (void)a;
    (void)b;
    return -1.0;
}

double distanceFFT(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    // TODO: implement — sqrt(sum((a_i - b_i)^2))
    (void)a;
    (void)b;
    return -1.0;
}

int classifyShape(
    const ShapeDescriptor& query,
    const std::vector<ShapeDescriptor>& refs,
    double huWeight,
    double fftWeight)
{
    // TODO: implement — weighted nearest-neighbor, return id of closest ref
    (void)query;
    (void)refs;
    (void)huWeight;
    (void)fftWeight;
    return -1;
}

} // namespace classify
