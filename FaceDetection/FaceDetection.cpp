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
    
    string outputFolder = "output";
        
    string model_file_d12, trained_file_d12;
    model_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection12.prototxt";
    trained_file_d12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection12_train_iter_298000.caffemodel";
    string model_file_c12, trained_file_c12;
    model_file_c12 = "/home/fanglin/caffe/FaceDetection/models/deploy_calibration12.prototxt";
    trained_file_c12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_calibration12_train_iter_410000.caffemodel";
    
    FaceClassifier detector_12(model_file_d12, trained_file_d12);
    FaceClassifier calibrator_12(model_file_c12, trained_file_c12);
    
    g_nWindows = 0;
    g_nDetectedWindows = 0;
    
    for (int i = 0; i < imgPaths.size(); i++) {
        string imgPath = imgPaths[i];
        Mat img = imread(imgPath);
        vector<Rect> resultRects;
        vector<float> scores;
        tic();
        FaceDetection(img, detector_12, calibrator_12, resultRects, scores);
        toc();
        for (int i = 0; i < resultRects.size(); i++) {
            cv::rectangle(img, resultRects[i], CV_RGB(255, 0, 0), 2);
        }
        
        stringstream ss;
        ss << outputFolder << "/orig_" << basename(imgPath.c_str());
        imwrite(ss.str(), img);
        imshow("img", img);
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
    }
    
    cout << "Average total windows " << g_nWindows*1.0/imgPaths.size() << endl;
    cout << "Average detected windows " << g_nDetectedWindows/imgPaths.size() << endl;
}
