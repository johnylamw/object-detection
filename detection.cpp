
#include <opencv2/opencv.hpp>
#include <shared_mat/shared_mat.h>
#include <stdio.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        // cout << "Usage: ./detection " << endl;
    }

    // MainReactor Setup:
    string colorMatName = "DAI_COLOR_0";
    string depthMatName = "DAI_DEPTH_0";
    
    SharedMat sharedColorMat(colorMatName.c_str());
    SharedMat sharedDepthMat(depthMatName.c_str());

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

        if (waitKey(1) == 27) {
            break;
        }
    }

}   