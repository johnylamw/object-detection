
#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
    Mat videoMat;
    namedWindow("Video Display");

    // Create video capture on camera port 0
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Cannot open camera";
    }

    // Display the video mat
    while (true) {
        cap >> videoMat;
        imshow("Video Display", videoMat);
        if ((char) 113 == (char) waitKey(1)) {
            break;
        }
    }
    return 0;
}