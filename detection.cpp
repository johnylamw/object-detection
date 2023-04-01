
#include <opencv2/opencv.hpp>
#include <shared_mat/shared_mat.h>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        // cout << "Usage: ./detection " << endl;
    }

    // Camera Calibration Parameters
    // TODO: implement parsing from json.
    float fx, fy, cx, cy, horizontalFOV, verticalFOV;
    fx = 616.4480388322528;
    fy = 616.2370137161736;
    cx = 428.36537439860047;
    cy = 247.20381979126174;
    horizontalFOV = 69;
    verticalFOV = 54;

    // MainReactor Setup:
    string colorMatName = "DAI_COLOR_0";
    string depthMatName = "DAI_DEPTH_0";
    
    SharedMat sharedColorMat(colorMatName.c_str());
    SharedMat sharedDepthMat(depthMatName.c_str());

    // Set up Blob Detection
    SimpleBlobDetector detector = *setupBlobDetection();
    
    // Looping through each iterations of frames
    while (1) {
        sharedColorMat.waitForFrame();
        sharedDepthMat.waitForFrame();

        Mat colorFrame = sharedColorMat.mat;
        Mat depthFrame = sharedDepthMat.mat;

        cv::imshow("Color Mat", colorFrame);
        cv::imshow("Depth Mat", depthFrame);

        Mat labFrame;
        cv::cvtColor(colorFrame, labFrame, ColorConversionCodes::COLOR_BGR2Lab);

        // Process Histogram here
        Mat histogram;

        // Process Backprojection
        Mat backProjectionFrame;
        int histogramChannels[] = {1, 2};
        float aRange[] = {0, 256};
        float bRange[] = {0, 256};
        float* histogramRanges[] = {aRange, bRange};
        cv::calcBackProject(&labFrame, 1, histogramChannels, histogram, backProjectionFrame, (const float**) histogramRanges);

        // Thresholding
        Mat threshold;
        Mat thresholdFrame;
        cv::threshold(backProjectionFrame, threshold, 35, 255, cv::THRESH_BINARY);
        cv::bitwise_not(threshold, thresholdFrame);
        
        // Morphological Transformation
        Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        Mat morphologyFrame;
        cv::morphologyEx(thresholdFrame, morphologyFrame, cv::MORPH_ELLIPSE, kernel, cv::Point(-1, -1), 4, 0 ,cv::morphologyDefaultBorderValue());

        // Detect Blobs
        std::vector<KeyPoint> keypoints;
        detector.detect(morphologyFrame, keypoints);
        
        Mat keypointsFrame;
        cv::drawKeypoints(colorFrame, keypoints, keypointsFrame, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        // Retrieve keypoints and process them
        if (keypoints.size() > 0) {
            double closestObjectDistance = 99999999;
            double closestObjectAngle = 99999999;
            
            for (int i = 0; i < keypoints.size(); i++) {
                KeyPoint point = keypoints[i];
                int x = point.pt.x;
                int y = point.pt.y;
                double depth = depthFrame.at<double>(y, x); // We have depth of our blobs :)

                // Get the bearing
                cv::Size frameSize = colorFrame.size();
                int width = frameSize.width;    // horizontal
                int height = frameSize.height;  // vertical
                float horizontalBearing = calculateBearing(width, horizontalFOV, cx, x);
                float verticalBearing = calculateBearing(height, verticalFOV, cy, y);

                if (depth < 250) {
                    // Set depth to 0 if it's less than 250mm
                    depth = 0;
                }

                float distance = calculateDistance(depth, horizontalBearing);

                // Draw a circle on the center of the blob
                int dot_size = (int) (point.size / 20);
                cv::circle(keypointsFrame, cv::Point(x, y), dot_size, cv::Scalar(255, 255, 255));

                // Add the depth and bearing of the showing
                string distanceDisplay = to_string(horizontalBearing) << "deg" << to_string(distance) << " mm";
                cv::putText(keypointsFrame, distanceDisplay, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 3, cv::LINE_AA, false);

                if (closestObjectDistance > distance) {
                    closestObjectDistance = distance;
                    closestObjectAngle = horizontalBearing;
                }
            }
        }

        // Contours: Draw Contours - IGNORE FOR NOW.

        if (waitKey(1) == 27) {
            break;
        }
    }
}   

Ptr<SimpleBlobDetector> setupBlobDetection() {
    cv::SimpleBlobDetector::Params params;
    
    params.minDistBetweenBlobs = 20;
    params.minThreshold = 0;
    params.maxThreshold = 10;
    params.thresholdStep = 10;
    params.minRepeatability = 1;

    params.filterByColor = true;
    params.blobColor = 0;
    
    params.filterByArea = true;
    params.minArea = 1000;
    params.maxArea = 240000;

    params.filterByCircularity = false;
    // params.minCircularity = 0.1
    // params.maxCircularity = 0.5 
    
    params.filterByInertia = false;
    // params.minInertiaRatio = 0.01
    // params.maxInertiaRatio = 0.80
    
    params.filterByConvexity = false;
    
    Ptr<SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
    
    return detector;
}

float calculateBearing(float size, float fov, float c, int centroid) {
    float ratio = fov/size;
    float diff, bearing;
    if (centroid > c) {
        // positive angle
        diff = centroid - c;
        bearing = diff * ratio;
    } else {
        // negative angle
        diff = c - centroid;
        bearing = -1 * diff * ratio;
    }

    return bearing;
}

float calculateDistance(int depth, float degree) {
    float pi = std::numbers::pi;
    float radians = (degree * (pi / 180.0));
    return depth/sin(90 - radians);
}