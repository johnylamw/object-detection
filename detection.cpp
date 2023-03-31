
#include <opencv2/opencv.hpp>
#include <shared_mat/shared_mat.h>
#include <stdio.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        // cout << "Usage: ./detection " << endl;
    }

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

        // im_with_keypoints = cv.drawKeypoints(color_frame, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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