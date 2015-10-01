#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

#include "FaceClassifier.h"
#include "FaceDetector.h"
#include "utilities_common.h"
#include "config.pb.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    
    int min_FaceSize; float scaleStep; int spacing;
    min_FaceSize = 28; scaleStep = 1.18; spacing = 4;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.LoadConfigs("/home/fanglin/caffe/FaceDetection/faceConfig_2nd.txt");       
    
    tic();
    VideoCapture vc;
    vc.open("/media/ssd/data/IDA Video Challenge/Stage 1/03 S11303.mp4");
    Mat frame;
    while (1) {
        vc >> frame;
        if (frame.empty()) {
            cout << "No frame" << endl;
            break;
        }
        
        vector<Rect> rects;
        vector<float> scores;
        int nWs = facedetector.Detect(frame, rects, scores);
        for (int i = 0; i < rects.size(); i++) {           
            //AFLWRect2FDDB(rects[i]);
            cv::rectangle(frame, rects[i], CV_RGB(255, 0, 0), 2);            
        }
        
        imshow("frame", frame);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
    }
    return 0;
}   



