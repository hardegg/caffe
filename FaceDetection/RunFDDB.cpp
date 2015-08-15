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

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::string;

int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    GenerateCalibLabelSet();
    Caffe::set_mode(Caffe::CPU);
    // Get files from FDDB
    string folderPath = "/home/fanglin/data/AFW";
    vector<string> imgPaths;
    GetFilePaths(folderPath, ".jpg", imgPaths);
    
    string outputFolder = "output_calib";
        
    string model_file_d12, trained_file_d12;
    model_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection12.prototxt";
    trained_file_d12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection12_train_iter_298000.caffemodel";
    string model_file_c12, trained_file_c12;
    model_file_c12 = "/home/fanglin/caffe/FaceDetection/models/deploy_calibration12.prototxt";
    trained_file_c12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_calibration12_train_iter_410000.caffemodel";
    
    FaceClassifier detector_12(model_file_d12, trained_file_d12);
    FaceClassifier calibrator_12(model_file_c12, trained_file_c12);

    string baseDir = "/media/ssd/data/FDDB";
    string listFilepath = baseDir + "/FDDB-folds/imgList.txt";
    string detFilepath = baseDir + "/FDDB-folds/detection_d12.txt";
    string annoFile = baseDir + "/FDDB-folds/annotation.txt";
    string imgDir = baseDir + "/originalPics";
    
    ifstream pathin;
    pathin.open(listFilepath.c_str());
    string t;
    
    ofstream of;
    of.open(detFilepath.c_str());
    long long nTotalWindows = 0;
    long long nDetected = 0;
    int nImages = 0;
    while(pathin >> t) {
        tic();
        ++nImages;
        Mat img = imread(imgDir + "/" + t + ".jpg");
        cout << t << endl;
        vector<Rect> rects;
        vector<float> scores;
        int nWs = FaceDetection(img, detector_12, calibrator_12, rects, scores);
        //int nWs = GetAllWindows(img, rects);
        nTotalWindows += nWs;
        nDetected += rects.size();
        
        for (int i = 0; i < rects.size(); i++) {
            // Expand by 20% vertically
            rects[i].y -= rects[i].height*0.1;
            rects[i].height *= 1.2;
            cv::rectangle(img, rects[i], CV_RGB(255, 0, 0), 2);            
        }
        
        of << t << endl;
        of << rects.size() << endl;
        for (int i = 0; i < rects.size(); i++) {
            of << rects[i].x << " " << rects[i].y << " " << rects[i].width << " "
                    << rects[i].height << " " << 1 << endl;
        }

        
//        imshow("img", img);
//        char c = cv::waitKey(1);
//        if (c == 'q')
//            break;q
        toc();
    }
    
    cout << "Average sliding windows " << nTotalWindows/nImages << endl;
    cout << "Average detected " << nDetected/nImages << endl;
    
    
    of.close();
    pathin.close();
    return 0;
}   


