#include "classification.hpp"
#include "image_procesing.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace classify {

std::vector<ShapeDescriptor> loadReferenceDescriptors(
    const std::string& refDir,
    int numShapes)
{
    // TODO: implement — for each Forma_XX.png: load, grayscale, threshold,
    //   findContours, extractLargestContour, computeHuMoments, computeFFTDescriptors
    (void)refDir;
    (void)numShapes;
    return {};
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
