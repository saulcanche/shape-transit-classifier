#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "image_procesing.hpp"
#include "classification.hpp"

// ============================================================================
// Helpers — create synthetic contours for testing
// ============================================================================

// Returns a rectangular contour centered at (cx, cy) with given width/height.
static std::vector<cv::Point> makeRect(int cx, int cy, int w, int h)
{
    int x0 = cx - w / 2, y0 = cy - h / 2;
    return {{x0, y0}, {x0 + w, y0}, {x0 + w, y0 + h}, {x0, y0 + h}};
}

// Draw a contour on a black image and return the binary image.
static cv::Mat drawContour(const std::vector<cv::Point>& contour, int rows, int cols)
{
    cv::Mat img = cv::Mat::zeros(rows, cols, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours = {contour};
    cv::drawContours(img, contours, 0, cv::Scalar(255), cv::FILLED);
    return img;
}

// ============================================================================
//  ImageProcessing Test Suite
// ============================================================================

TEST(ImageProcessing, ExtractLargestContour_PicksBiggest)
{
    // Create two contours: small (20x20) and large (100x100)
    auto small_rect = makeRect(50,  50,  20, 20);
    auto large_rect = makeRect(200, 200, 100, 100);

    std::vector<std::vector<cv::Point>> contours = {small_rect, large_rect};
    auto result = imgproc::extractLargestContour(contours);

    // The largest contour should be the 100x100 one
    ASSERT_FALSE(result.empty());
    double area = cv::contourArea(result);
    EXPECT_GT(area, cv::contourArea(small_rect));
}

TEST(ImageProcessing, ExtractLargestContour_EmptyInput)
{
    std::vector<std::vector<cv::Point>> empty;
    auto result = imgproc::extractLargestContour(empty);
    EXPECT_TRUE(result.empty());
}

TEST(ImageProcessing, ComputeCentroid_CenteredRect)
{
    // Rectangle centered at (150, 100)
    auto rect = makeRect(150, 100, 60, 40);
    cv::Point2f centroid = imgproc::computeCentroid(rect);

    EXPECT_NEAR(centroid.x, 150.0, 5.0);
    EXPECT_NEAR(centroid.y, 100.0, 5.0);
}

TEST(ImageProcessing, IsAtCenter_True)
{
    cv::Point2f centroid(320.0f, 240.0f);
    int frameWidth = 640;
    int tolerance  = 20;

    EXPECT_TRUE(imgproc::isAtCenter(centroid, frameWidth, tolerance));
}

TEST(ImageProcessing, IsAtCenter_False)
{
    cv::Point2f centroid(50.0f, 240.0f);
    int frameWidth = 640;
    int tolerance  = 20;

    EXPECT_FALSE(imgproc::isAtCenter(centroid, frameWidth, tolerance));
}

TEST(ImageProcessing, ComputeHuMoments_SevenValues)
{
    auto rect = makeRect(100, 100, 80, 60);
    auto hu = imgproc::computeHuMoments(rect);

    ASSERT_EQ(hu.size(), 7u);
    // At least one Hu moment should be non-zero for a real contour
    bool anyNonZero = false;
    for (double v : hu) {
        if (std::abs(v) > 1e-15) anyNonZero = true;
    }
    EXPECT_TRUE(anyNonZero);
}

TEST(ImageProcessing, ComputeHuMoments_Invariance)
{
    // Same-sized rectangle at two different positions should have similar Hu moments
    auto rectA = makeRect(100, 100, 80, 60);
    auto rectB = makeRect(300, 250, 80, 60);

    auto huA = imgproc::computeHuMoments(rectA);
    auto huB = imgproc::computeHuMoments(rectB);

    for (int i = 0; i < 7; ++i) {
        // Log-scale comparison — they should be very close for translated shapes
        if (std::abs(huA[i]) > 1e-15 && std::abs(huB[i]) > 1e-15) {
            double diff = std::abs(std::log(std::abs(huA[i])) -
                                   std::log(std::abs(huB[i])));
            EXPECT_LT(diff, 0.5) << "Hu moment " << i << " differs too much";
        }
    }
}

TEST(ImageProcessing, ContourToComplex_SameLength)
{
    auto rect = makeRect(100, 100, 60, 40);
    cv::Point2f centroid = imgproc::computeCentroid(rect);
    // NOTE: this test will only be meaningful after computeCentroid is implemented.
    // For now, use a known centroid.
    cv::Point2f knownCentroid(100.0f, 100.0f);

    auto sig = imgproc::contourToComplexSignature(rect, knownCentroid);
    EXPECT_EQ(sig.size(), rect.size());
}

TEST(ImageProcessing, ComputeFFTDescriptors_CorrectLength)
{
    // Build a simple complex signature manually
    std::vector<std::complex<double>> sig;
    for (int i = 0; i < 64; ++i) {
        double angle = 2.0 * M_PI * i / 64.0;
        sig.emplace_back(std::cos(angle), std::sin(angle));
    }

    int N = 16;
    auto desc = imgproc::computeFFTDescriptors(sig, N);
    EXPECT_EQ(static_cast<int>(desc.size()), N);
}

TEST(ImageProcessing, ComputeFFTDescriptors_Normalized)
{
    // Circle-like signature: all magnitudes equal
    std::vector<std::complex<double>> sig;
    for (int i = 0; i < 64; ++i) {
        double angle = 2.0 * M_PI * i / 64.0;
        double r = 50.0;
        sig.emplace_back(r * std::cos(angle), r * std::sin(angle));
    }

    auto desc = imgproc::computeFFTDescriptors(sig, 16);
    // After normalization by DC component, the first value should be 1.0
    ASSERT_FALSE(desc.empty());
    EXPECT_NEAR(desc[0], 1.0, 0.01);
}

// ============================================================================
//  Classification Test Suite
// ============================================================================

TEST(Classification, DistanceHuMoments_IdenticalIsZero)
{
    std::array<double, 7> hu = {1.5e-1, 3.2e-3, 7.8e-5, 1.1e-5, -2.3e-10, 4.5e-7, 1.2e-10};
    double d = classify::distanceHuMoments(hu, hu);
    EXPECT_DOUBLE_EQ(d, 0.0);
}

TEST(Classification, DistanceHuMoments_DifferentPositive)
{
    std::array<double, 7> a = {1.5e-1, 3.2e-3, 7.8e-5, 1.1e-5, 2.3e-10, 4.5e-7, 1.2e-10};
    std::array<double, 7> b = {2.0e-1, 1.0e-3, 5.0e-5, 3.0e-5, 1.0e-10, 9.0e-7, 3.0e-10};
    double d = classify::distanceHuMoments(a, b);
    EXPECT_GT(d, 0.0);
}

TEST(Classification, DistanceFFT_IdenticalIsZero)
{
    std::vector<double> v = {1.0, 0.5, 0.3, 0.1, 0.05};
    double d = classify::distanceFFT(v, v);
    EXPECT_DOUBLE_EQ(d, 0.0);
}

TEST(Classification, DistanceFFT_DifferentPositive)
{
    std::vector<double> a = {1.0, 0.5, 0.3};
    std::vector<double> b = {1.0, 0.8, 0.1};
    double d = classify::distanceFFT(a, b);
    EXPECT_GT(d, 0.0);
}

TEST(Classification, ClassifyShape_SelfMatch)
{
    // Build two fake reference descriptors
    classify::ShapeDescriptor refA;
    refA.id = 0;
    refA.huMoments = {1.5e-1, 3.2e-3, 7.8e-5, 1.1e-5, 2.3e-10, 4.5e-7, 1.2e-10};
    refA.fftDescriptors = {1.0, 0.5, 0.3, 0.1};

    classify::ShapeDescriptor refB;
    refB.id = 1;
    refB.huMoments = {2.0e-1, 1.0e-3, 5.0e-5, 3.0e-5, 1.0e-10, 9.0e-7, 3.0e-10};
    refB.fftDescriptors = {1.0, 0.8, 0.1, 0.05};

    std::vector<classify::ShapeDescriptor> refs = {refA, refB};

    // Query is identical to refA — should classify as id 0
    int result = classify::classifyShape(refA, refs, 1.0, 1.0);
    EXPECT_EQ(result, 0);
}

TEST(Classification, LoadReferenceDescriptors_Count)
{
    // This test loads the actual reference images from data/reference_shapes/
    // It will only pass once loadReferenceDescriptors AND the imgproc functions are implemented.
    std::string refDir = "data/reference_shapes";
    auto descs = classify::loadReferenceDescriptors(refDir, 17);
    EXPECT_EQ(static_cast<int>(descs.size()), 17);
}
