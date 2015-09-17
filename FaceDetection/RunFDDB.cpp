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

    string baseDir = "/media/ssd/data/FDDB";
    string listFilepath = baseDir + "/FDDB-folds/imgList.txt";
    string detFilepath = baseDir + "/FDDB-folds/detection_d12.txt";
    string annoFile = baseDir + "/FDDB-folds/annotation.txt";
    string imgDir = baseDir + "/originalPics";
    
    ifstream pathin, annoin;
    pathin.open(listFilepath.c_str());
    //annoin.open(annoFile.c_str());
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
        int nWs = facedetector.Detect(img, rects, scores);
        //int nWs = facedetector.GetSlidingWindows(img, rects, scores);
        
        cout << "Total sliding windows " << nWs << endl;
        cout << "Detected faces " << rects.size() << endl; 
        //int nWs = GetAllWindows(img, rects);
        nTotalWindows += nWs;
        nDetected += rects.size();
        
        for (int i = 0; i < rects.size(); i++) {           
            AFLWRect2FDDB(rects[i]);
            cv::rectangle(img, rects[i], CV_RGB(255, 0, 0), 2);            
        }
        
        of << t << endl;
        of << rects.size() << endl;
        
        for (int i = 0; i < rects.size(); i++) {
            of << rects[i].x << " " << rects[i].y << " " << rects[i].width << " "
                    << rects[i].height << " " << scores[i] << endl;
        }
        
        
//        imshow("img", img);
//        char c = cv::waitKey(0);
//        if (c == 'q')
//            break;
        toc();
    }
    
    cout << "Average sliding windows " << 1.0*nTotalWindows/nImages << endl;
    cout << "Average detected " << 1.0f*nDetected/nImages << endl;
    
    
    of.close();
    pathin.close();
    return 0;
}   


