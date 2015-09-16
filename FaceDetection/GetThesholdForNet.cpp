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
    bool BATCH = true;
    
    string model_file_d12, trained_file_d12, mean_file_d12;
    model_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy/deploy_detection12.prototxt";
    trained_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy/facecascade_detection12_train_iter_66000.caffemodel";
    mean_file_d12 = "/home/fanglin/caffe/FaceDetection/models/deploy/12net_mean_const128.binaryproto";
    
    
    FaceClassifier detector_12(model_file_d12, trained_file_d12, mean_file_d12);
    
    string model_file_d24, trained_file_d24;
    model_file_d24 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection24.prototxt";
    trained_file_d24 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection24_train_iter_500000.caffemodel";
    //FaceClassifier detector_24(model_file_d24, trained_file_d24);

    
    string model_file_d48, trained_file_d48;
    model_file_d48 = "/home/fanglin/caffe/FaceDetection/models/deploy_detection48.prototxt";
    trained_file_d48 = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection48_train_iter_470000.caffemodel";
    //FaceClassifier detector_48(model_file_d48, trained_file_d12);
    
    string listFilepath = "/media/ssd/data/aflw/data/faces/detection12_val.txt";
    //string listFilepath = "/media/ssd/data/aflw/data/neg48_train.txt";

    
    ifstream pathin;
    pathin.open(listFilepath.c_str());
    string line;
    
    vector<float> scores;
    int nCorrect = 0;
    int nImages = 0;
    int batchSize = 256;
    vector<Mat> imgs;
    while(std::getline(pathin, line)) {
        string filepath =  line.substr(0, line.find(' '));
        Mat img = imread(filepath);
        Mat rsz;
        cv::resize(img, rsz, Size(12, 12));
        img = rsz;
        if (img.empty())
            break;
        if (BATCH) {
            imgs.push_back(img);
            std::vector<float> dscores;
            if ((nImages+1) % batchSize == 0) {           
                dscores  = detector_12.Predict(imgs);            
                for (int i = 0; i < imgs.size(); i++) {
                    if (dscores[2*i+1] > dscores[2*i])
                        ++nCorrect;
                    scores.push_back(dscores[2*i+1]);                   
                }
               imgs.clear();
            }
        } else {
            vector<float> dscores  = detector_12.Predict(img);
            //vector<Prediction> prdct = detector_12.Classify(img);

            //if (prdct[0].first == 1)
            if (dscores[1] > dscores[0])
                ++nCorrect;     
            scores.push_back(dscores[1]);
        }
        //imshow("img", img);
        char c = cv::waitKey(1);
        if (c == 'q')
            break;
        ++nImages;
        if (nImages %10000 == 0)
            cout << nImages << endl;
        
        
    }
    
    // Process the left images
    if (BATCH & imgs.size() > 0) {
        vector<float> dscores  = detector_12.Predict(imgs);
            
        for (int i = 0; i < imgs.size(); i++) {
            if (dscores[2*i+1] > dscores[2*i])
                ++nCorrect;
            scores.push_back(dscores[2*i+1]);                   
        }
    }
    
    std::sort(scores.begin(), scores.end());
    float threshold = 0.99;
    int pos = (1-threshold)*scores.size();
    cout << nImages << " images" << endl;
    cout << "accuracy = " << 1.0*nCorrect/nImages << endl;
    cout << "threshold = " << scores[pos] << endl;
    return 0;
}
