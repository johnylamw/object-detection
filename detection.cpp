
#include <opencv2/opencv.hpp>
#include <shared_mat/shared_mat.h>
#include <stdio.h>
#include <math.h>
#include <array>

#include "networktables/NetworkTableInstance.h"
#include <networktables/DoubleTopic.h>
#include <networktables/StringTopic.h>

using namespace cv;
using namespace std;

float fx, fy, cx, cy, horizontalFOV, verticalFOV;
float link1, link2;
std::shared_ptr<nt::NetworkTable> blackMesaTable;
std::shared_ptr<nt::NetworkTable> armTable;

int main(int argc, char** argv) {
    if (argc == 1 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        // cout << "Usage: ./detection " << endl;
    }
    // Setup NetworkTables
    nt::NetworkTableInstance instance = nt::NetworkTableInstance::GetDefault();
    instance.StartClient4("detection");
    instance.SetServerTeam(488);

    blackMesaTable = instance.GetTable("SmartDashboard")->GetSubTable("BlackMesa");
    armTable = instance.GetTable("SmartDashboard")->GetSubTable("UnifiedArmSubsystem");

    // Camera Calibration Parameters
    // TODO: implement parsing from json.
    fx = 616.4480388322528;
    fy = 616.2370137161736;
    cx = 428.36537439860047;
    cy = 247.20381979126174;
    horizontalFOV = 69;
    verticalFOV = 54;
    link1 = 100;
    link2 = 100;

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
        bool objectsIdentified = keypoints.size() > 0;
        if (objectsIdentified) {
            double closestObjectDistance = 99999999;
            double closestObjectAngle = 99999999;
            float closestRobotToObjectDistance;
            
            for (int i = 0; i < keypoints.size(); i++) {
                KeyPoint point = keypoints[i];
                int x = point.pt.x;
                int y = point.pt.y;
                int depth = depthFrame.at<int>(y, x); // We have depth of our blobs :)

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
                float robotToObjectDistance;
                float* XYZ = calculateObjectXYZ(colorFrame, x, y, distance, &robotToObjectDistance);


                // Draw a circle on the center of the blob
                int dot_size = (int) (point.size / 20);
                cv::circle(keypointsFrame, cv::Point(x, y), dot_size, cv::Scalar(255, 255, 255));

                // Add the depth and bearing of the showing
                string distanceDisplay = to_string(horizontalBearing) << "deg" << to_string(distance) << " mm";
                cv::putText(keypointsFrame, distanceDisplay, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 3, cv::LINE_AA, false);
                if (closestObjectDistance > distance) {
                    closestObjectDistance = distance;
                    closestRobotToObjectDistance = robotToObjectDistance;
                    closestObjectAngle = horizontalBearing;
                }
            }
            // Update NT w/ Object State
            blackMesaTable->GetEntry("cameraToObjectDistance").SetFloat(closestObjectDistance, nt::Now());
            blackMesaTable->GetEntry("robotToObjectDistance").SetFloat(closestRobotToObjectDistance, nt::Now());
            blackMesaTable->GetEntry("angle").SetFloat(closestObjectAngle, nt::Now());
        }

        // Update identification state:
        blackMesaTable->GetEntry("found").SetBoolean(objectsIdentified, nt::Now());
        blackMesaTable->GetEntry("targetNum").SetInteger(keypoints.size(), nt::Now());

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

float degreeToRadians(float degree) {
    float pi = std::numbers::pi;
    float radians = (degree * (pi / 180.0));
    return radians;
}

float calculateDistance(int depth, float degree) {
    float radians = degreeToRadians(degree);
    return depth/sin(90 - radians);
}

float* calculateObjectXYZ(Mat frame, int u, int v, int depth, float* resultDistance) {
    // Image coordinates
    cv::Size frameSize = frame.size();
    float x = u/frameSize.width;
    float y = v/frameSize.height;

    // Convert to camera coordinates
    float Xc = (x - cx) / fx;
    float Yc = (y - cy) / fy;
    float Zc = 1.0;
    vector<float> XYZc{Xc * depth, Yc * depth, Zc * depth};

    // Convert to world coordinates in robot space
    // Retrieve the arm angles via NetworkTables
    // TODO: might be a double array?
    double lowerArmPosition = armTable->GetSubTable("LowerArm")->GetEntry("AbsoluteEncoderPosition").GetDouble(-1.0);
    double upperArmPosition = armTable->GetSubTable("LowerArm")->GetEntry("AbsoluteEncoderPosition").GetDouble(-1.0);
    vector<float> translationVector = {};
    float rotationalMatrix[3][3] = {};

    forwardKinematicsSolver(link1, link2, degreeToRadians(lowerArmPosition), degreeToRadians(upperArmPosition), translationVector, rotationalMatrix);

    // Dot product and adding to translational vector to get the real world coordinates:
    float XYZw[3];
    for (int i = 0; i < sizeof(rotationalMatrix); i++) {
       XYZw[i] = translationVector[i];
            for (int j = 0; j < sizeof(rotationalMatrix[i]); j++) {
            XYZw[i] += rotationalMatrix[i][j] * XYZc[j];
        }
    }
    // Calculate the distance
    float distance = sqrt(pow(XYZw[0], 2) + pow(XYZw[1], 2) + pow(XYZw[2], 2));
    *resultDistance = distance;

    return XYZw;
}

void forwardKinematicsSolver(float L1, float L2, float theta1, float theta2, vector<float> tVec, float rotMat[][3]) {
    float end_effector_transformation[4][4] = {
        {std::cos(theta1 + theta2), -std::sin(theta1 + theta2), 0, L1 * std::cos(theta1) + L2 * std::cos(theta1 + theta2)},
        {std::sin(theta1 + theta2), std::cos(theta1 + theta2), 0, L1 * std::sin(theta1) + L2 * std::sin(theta1 + theta2)},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };

    // Find the actual rotation and translation
    // We need a rotational and translation matrix as respect to arm to camera
    // 90 degrees around the x-axis
    float camera_rotation_matrix[4][4] = {
        {1, 0, 0, 0},
        {0, 0, -1, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1}
    };
    
    // 0.1 meter along the z-axis of the arm
    float camera_translation_matrix[4][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0.1},
        {0, 0, 0, 1}
    };
    
    // Camera to end-effector tranformations
    float final_transformation[4][4];
    multiplyMatrices(end_effector_transformation, camera_rotation_matrix, final_transformation);
    multiplyMatrices(final_transformation, camera_translation_matrix, final_transformation);
    
    // Extract the rotation matrix and translation vector
    array<array<float, 3>, 3> rot_mat = {{
        {final_transformation[0][0], final_transformation[0][1], final_transformation[0][2]},
        {final_transformation[1][0], final_transformation[1][1], final_transformation[1][2]},
        {final_transformation[2][0], final_transformation[2][1], final_transformation[2][2]}
    }};

    vector<float> tran_vector{final_transformation[0][3], final_transformation[1][3], final_transformation[2][3]};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rot_mat[i][j] = final_transformation[i][j];
        }
    }
}

// Multiplies two 4x4 matrices
void multiplyMatrices(float A[][4], float B[][4], float result[][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}