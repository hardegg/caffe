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
    int net = 24;
    float threshold = 0.99;
    
    string model_file, trained_file, mean_file;
    vector<string> mean_files;
    string listFilepath;
    if (net == 12) {
        threshold = 0.99;
        model_file = "/home/fanglin/caffe/FaceDetection/models/deploy/deploy_detection12.prototxt";
        trained_file = "/home/fanglin/caffe/FaceDetection/models/deploy/facecascade_detection12_train_iter_59000.caffemodel";
        mean_file = "/home/fanglin/caffe/FaceDetection/models/deploy/12net_mean_const128.binaryproto";
        listFilepath = "/media/ssd/data/aflw/data/faces/detection12_val_noflip.txt";
    }
    
    if (net == 24) {
        threshold = 0.97;
        model_file = "/home/fanglin/caffe/FaceDetection/models/deploy/deploy_detection24_with12.prototxt";
        trained_file = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection24_with12_train_iter_39000.caffemodel";
        mean_file = "/home/fanglin/caffe/FaceDetection/models/deploy/24net_mean_const128.binaryproto";
        listFilepath = "/media/ssd/data/aflw/data/faces/detection24_val_noflip.txt";
        //listFilepath = "/media/ssd/data/aflw/data/neg24_val.txt";

        mean_files.push_back("/home/fanglin/caffe/FaceDetection/models/deploy/12net_mean_const128.binaryproto");
        mean_files.push_back(mean_file);
    }
    
    if (net == 48) {
        threshold = 0.97;
        model_file = "/home/fanglin/caffe/FaceDetection/models/deploy/deploy_detection48.prototxt";
        trained_file = "/home/fanglin/caffe/FaceDetection/models/snapshots/facecascade_detection48_train_iter_59000.caffemodel";
        mean_file = "/home/fanglin/caffe/FaceDetection/models/deploy/48net_mean_const128.binaryproto";
        listFilepath = "/media/ssd/data/aflw/data/faces/detection48_val_noflip.txt";
    }
    
    
    FaceClassifier detector(model_file, trained_file, mean_files);  
    
    
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
        cv::resize(img, rsz, Size(net, net));
        
        img = rsz;
        if (img.empty())
            break;
        if (BATCH) {
            imgs.push_back(img);
            std::vector<float> dscores;
            if ((nImages+1) % batchSize == 0) {           
                dscores  = detector.Predict(imgs);            
                for (int i = 0; i < imgs.size(); i++) {
                    if (dscores[2*i+1] > dscores[2*i]) {
                        //cout << dscores[2*i+1] << ", " << dscores[2*i] << endl;
                        ++nCorrect;
                    }
                    scores.push_back(dscores[2*i+1]);                   
                }
               imgs.clear();
            }
        } else {
            vector<float> dscores  = detector.Predict(img);
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
        vector<float> dscores  = detector.Predict(imgs);
            
        for (int i = 0; i < imgs.size(); i++) {
            if (dscores[2*i+1] > dscores[2*i])
                ++nCorrect;
            scores.push_back(dscores[2*i+1]);                   
        }
    }
    
    std::sort(scores.begin(), scores.end());
    
    int pos = (1-threshold)*scores.size();
    cout << nImages << " images" << endl;
    cout << "accuracy = " << 1.0*nCorrect/nImages << endl;
    cout << "threshold = " << scores[pos] << " for recall " << threshold << endl;
    return 0;
}
