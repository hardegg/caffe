#include "FaceClassifier.h"
#include "utilities_common.h"
#include <string>
#include <algorithm>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;

int main(int argc, const char* argv[])
{
    ::google::InitGoogleLogging(argv[0]);
    //Caffe::set_mode(Caffe::CPU);
    
    string model_file_d12, trained_file_d12;
    model_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection12.prototxt";
    trained_file_d12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection12_train_iter_160000.caffemodel";
    //FaceClassifier detector_12(model_file_d12, trained_file_d12);
    
    string model_file_d24, trained_file_d24;
    model_file_d24 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection24.prototxt";
    trained_file_d24 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection24_train_iter_500000.caffemodel";
    FaceClassifier detector_24(model_file_d24, trained_file_d24);

    
    string model_file_d48, trained_file_d48;
    model_file_d48 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection48.prototxt";
    trained_file_d48 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection48_train_iter_470000.caffemodel";
    FaceClassifier detector_48(model_file_d48, trained_file_d48);
    
    //string listFilepath = "/media/ssd/data/aflw/data/faces/detection48_train.txt";
    string listFilepath = "/media/ssd/data/aflw/data/neg48_train.txt";

    
    ifstream pathin;
    pathin.open(listFilepath.c_str());
    string line;
    
    vector<float> scores;
    int nCorrect = 0;
    int nImages = 0;
    while(std::getline(pathin, line)) {
        string filepath =  line.substr(0, line.find(' '));
        Mat img = imread(filepath);
//        Mat rsz;
//        cv::resize(img, rsz, Size(24, 24));
//        img = rsz;
        if (img.empty())
            break;
        
        //std::vector<float> dscores = detector_48.Predict(img);
        vector<Prediction> prdct = detector_48.Classify(img);
        if (prdct[0].first == 1)
            ++nCorrect;        
        //imshow("img", img);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
        ++nImages;
        if (nImages %10000 == 0)
            cout << nImages << endl;
    }
    
    std::sort(scores.begin(), scores.end());
    cout << "accuracy = " << 1.0*nCorrect/nImages << endl;
    return 0;
}
