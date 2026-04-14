#include "image_procesing.hpp"
#include "classification.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

int main()
{
    // --- Configuration ---
    const std::string videoPath    = "data/video.mpg";
    const std::string refDir       = "data/reference_shapes";
    const int         numShapes    = 17;
    const int         tolerance    = 20;
    const int         numFFTDesc   = 100;
    const int         resampleN    = 256;
    const double      huWeight     = 1.0;
    const double      fftWeight    = 5.0;
    const int         threshVal    = 128;
    const double      minArea      = 500.0;    // ignore tiny noise contours
    const double      matchDist    = 60.0;     // same-shape centroid matching
    const int         panelW       = 250;      // right panel width for reference img
    const int         panelPad     = 20;       // padding inside the panel

    // --- Load reference descriptors ---
    std::vector<classify::ShapeDescriptor> refs =
        classify::loadReferenceDescriptors(refDir, numShapes);

    if (refs.empty()) {
        std::cerr << "Error: no reference descriptors loaded." << std::endl;
        return 1;
    }

    // --- Pre-load reference images for display ---
    std::vector<cv::Mat> refImages(numShapes);
    for (int i = 0; i < numShapes; ++i) {
        std::string fn = refDir + "/Forma_" + (i < 10 ? "0" : "") + std::to_string(i) + ".png";
        refImages[i] = cv::imread(fn, cv::IMREAD_COLOR);
    }

    // --- Open video ---
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video " << videoPath << std::endl;
        return 1;
    }

    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));

    std::ofstream outFile("data/actual_list.txt");
    if (!outFile.is_open()) return 1;

    // --- State ---
    std::vector<cv::Point2f> prevCenterCentroids;
    int lastDetectedId = -1;          // last classified shape id
    cv::Mat lastRefDisplay;           // cached panel image of last detection

    cv::Mat frame, gray, binary;

    int visualDelay = 50;
    cv::namedWindow("Shape Transit Classifier", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Delay (ms)", "Shape Transit Classifier", &visualDelay, 200);

    while (cap.read(frame)) {
        // 1. Preprocess
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, binary, threshVal, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_NONE);

        // 2. Collect all contours currently at center
        std::vector<cv::Point2f> curCenterCentroids;
        std::vector<std::vector<cv::Point>> curCenterContours;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < minArea) continue;

            cv::Point2f centroid = imgproc::computeCentroid(contour);
            if (imgproc::isAtCenter(centroid, frameWidth, tolerance)) {
                curCenterCentroids.push_back(centroid);
                curCenterContours.push_back(contour);
            }
        }

        // 3. For each shape at center, check if it's NEW (no match in prev frame)
        for (size_t i = 0; i < curCenterCentroids.size(); ++i) {
            bool isNew = true;
            for (const auto& prev : prevCenterCentroids) {
                double dx = curCenterCentroids[i].x - prev.x;
                double dy = curCenterCentroids[i].y - prev.y;
                if (std::sqrt(dx * dx + dy * dy) < matchDist) {
                    isNew = false;
                    break;
                }
            }

            if (isNew) {
                // New shape just entered center — classify it
                std::vector<cv::Point> resampled =
                    imgproc::resampleContour(curCenterContours[i], resampleN);

                cv::Point2f rCentroid = imgproc::computeCentroid(resampled);
                std::array<double, 7> hu = imgproc::computeHuMoments(resampled);
                std::vector<std::complex<double>> sig =
                    imgproc::contourToComplexSignature(resampled, rCentroid);
                std::vector<double> fft =
                    imgproc::computeFFTDescriptors(sig, numFFTDesc);

                classify::ShapeDescriptor query;
                query.id = -1;
                query.huMoments = hu;
                query.fftDescriptors = fft;

                int bestId = classify::classifyShape(query, refs, huWeight, fftWeight);
                std::cout << "figId =  " << bestId << std::endl;
                outFile << "figId =  " << bestId << std::endl;
                lastDetectedId = bestId;

                // Prepare reference image for the side panel
                if (bestId >= 0 && bestId < numShapes && !refImages[bestId].empty()) {
                    // Fit reference image into the panel
                    cv::Mat ref = refImages[bestId];
                    int dispH = panelW - 2 * panelPad;
                    int dispW = dispH * ref.cols / ref.rows;
                    if (dispW > panelW - 2 * panelPad) {
                        dispW = panelW - 2 * panelPad;
                        dispH = dispW * ref.rows / ref.cols;
                    }
                    cv::resize(ref, lastRefDisplay, cv::Size(dispW, dispH));
                }
            }
        }

        // --- Draw visualization ---
        cv::Mat display;
        cv::Mat panel(frame.rows, panelW, CV_8UC3, cv::Scalar(30, 30, 30));

        // Draw detected contours on the frame (green outline)
        for (size_t i = 0; i < curCenterContours.size(); ++i) {
            cv::drawContours(frame, curCenterContours, static_cast<int>(i),
                             cv::Scalar(0, 255, 0), 2);
        }

        // Side panel: show last detected reference image + label
        if (lastDetectedId >= 0 && !lastRefDisplay.empty()) {
            int yOff = panelPad;
            // Title
            cv::putText(panel, "Detected:", cv::Point(panelPad, yOff + 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
            yOff += 35;
            // Shape id
            std::string label = "Forma " + std::to_string(lastDetectedId);
            cv::putText(panel, label, cv::Point(panelPad, yOff + 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 120), 2);
            yOff += 35;
            // Reference image
            if (yOff + lastRefDisplay.rows < panel.rows - panelPad) {
                int xOff = (panelW - lastRefDisplay.cols) / 2;
                lastRefDisplay.copyTo(
                    panel(cv::Rect(xOff, yOff,
                                   lastRefDisplay.cols, lastRefDisplay.rows)));
            }
        } else {
            cv::putText(panel, "Waiting...", cv::Point(panelPad, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(120, 120, 120), 1);
        }

        cv::hconcat(frame, panel, display);
        cv::imshow("Shape Transit Classifier", display);
        // Wait according to the trackbar delay, but ensure it's at least 1ms to prevent infinite pausing 
        if (cv::waitKey(std::max(1, visualDelay)) == 27) break;  // ESC to quit

        prevCenterCentroids = curCenterCentroids;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
