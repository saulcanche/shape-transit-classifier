#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <array>
#include <string>
#include <vector>

namespace classify {

// Holds the descriptor data for a single shape.
struct ShapeDescriptor {
    int id;
    std::array<double, 7> huMoments;
    std::vector<double> fftDescriptors;
};

// Loads reference images (Forma_00.png .. Forma_{numShapes-1}.png) from refDir,
// computes Hu moments and FFT descriptors for each, and returns them.
std::vector<ShapeDescriptor> loadReferenceDescriptors(
    const std::string& refDir,
    int numShapes);

// Computes the log-space L1 distance between two Hu moment vectors:
//   sum of |log|a_i| - log|b_i|| for i = 0..6
double distanceHuMoments(
    const std::array<double, 7>& a,
    const std::array<double, 7>& b);

// Computes the Euclidean distance between two FFT descriptor vectors.
double distanceFFT(
    const std::vector<double>& a,
    const std::vector<double>& b);

// Classifies a query shape against reference descriptors using a weighted
// combination of Hu-moment distance and FFT distance.
// Returns the id of the closest reference shape.
int classifyShape(
    const ShapeDescriptor& query,
    const std::vector<ShapeDescriptor>& refs,
    double huWeight,
    double fftWeight);

} // namespace classify

#endif // CLASSIFICATION_HPP
