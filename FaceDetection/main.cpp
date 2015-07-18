#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, const char *argv[])
{
    VideoCapture vc;
    vc.open("/home/fanglin/caffe/FaceDetection/20141210020001.mp4");
    Mat frame;
    while (1) {
        vc >> frame;
        if (frame.empty()) {
            cout << "No frame" << endl;
            break;
        }
        imshow("frame", frame);
        cv::waitKey(1);
    }
        
    cout << "Hello!" << endl;
    return 0;
}
