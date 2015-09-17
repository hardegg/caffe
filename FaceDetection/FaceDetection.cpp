#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FaceDetector.h"
#include "utilities_common.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

long g_nWindows;
long g_nDetectedWindows;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    GenerateCalibLabelSet();
    Caffe::set_mode(Caffe::CPU);
    // Get files from FDDB
    string folderPath = "/home/fanglin/data/AFW";
    vector<string> imgPaths;
    GetFilePaths(folderPath, ".jpg", imgPaths);
    
    string outputFolder = "output_calib";
    
     int min_FaceSize; float scaleStep; int spacing;
    min_FaceSize = 80; scaleStep = 1.414; spacing = 4;
    FaceDetector facedetector(min_FaceSize, scaleStep, spacing);
    facedetector.LoadConfigs("/home/fanglin/caffe/FaceDetection/faceConfig_2nd.txt");    
       
    g_nWindows = 0;
    g_nDetectedWindows = 0;
    
    for (int i = 0; i < imgPaths.size(); i++) {
        string imgPath = imgPaths[i];
        Mat img = imread(imgPath);
        vector<Rect> resultRects;
        vector<float> scores;
        tic();
        facedetector.Detect(img, resultRects, scores);
        toc();
        for (int i = 0; i < resultRects.size(); i++) {
            cv::rectangle(img, resultRects[i], CV_RGB(255, 0, 0), 2);
        }
        
        stringstream ss;
        ss << outputFolder << "/" << basename(imgPath.c_str());
        imwrite(ss.str(), img);
        imshow("img", img);
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
    }
    
    cout << "Average total windows " << g_nWindows*1.0/imgPaths.size() << endl;
    cout << "Average detected windows " << g_nDetectedWindows/imgPaths.size() << endl;
}
