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
    Caffe::set_mode(Caffe::CPU);
    
    string model_file_d12, trained_file_d12;
    model_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection12.prototxt";
    trained_file_d12 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection12_train_iter_160000.caffemodel";
    FaceClassifier detector_12(model_file_d12, trained_file_d12);
    
    string listFilepath = "/media/ssd/data/aflw/data/faces/detection12_val.txt";
    
    ifstream pathin;
    pathin.open(listFilepath.c_str());
    string line;
    
    vector<float> scores;
    int nCorrect = 0;
    while(std::getline(pathin, line)) {
        string filepath =  line.substr(0, line.find(' '));
        Mat img = imread(filepath);
        if (img.empty())
            break;
        
        std::vector<float> dscores = detector_12.Predict(img);
        vector<Prediction> prdct = detector_12.Classify(img);
        if (prdct[0].first == 1)
            ++nCorrect;
        scores.push_back(dscores[1]);
        imshow("img", img);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
    }
    
    std::sort(scores.begin(), scores.end());
    cout << "accuracy = " << 1.0*nCorrect/scores.size() << endl;
    return 0;
}
