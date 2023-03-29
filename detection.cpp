
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

        Mat colorMat = sharedColorMat.mat;
        Mat depthMat = sharedDepthMat.mat;

        imshow("Color Mat", colorMat);
        imshow("Depth Mat", depthMat);

        if (waitKey(1) == 27) {
            break;
        }
    }

}