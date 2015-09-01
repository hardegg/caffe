/* 
 * File:   FaceDetector.h
 * Author: fanglin
 *
 * Created on August 11, 2015, 6:38 PM
 */

#ifndef FACEDETECTOR_H
#define	FACEDETECTOR_H
#include <opencv2/opencv.hpp>
#include "FaceClassifier.h"
#include "config.pb.h"


using namespace cv;

void GenerateCalibLabelSet();
Rect CalibrateRect(const Rect rect, float ex, float ey, float es);
int FaceDetection(const Mat& img, FaceClassifier& detector12, FaceClassifier& detector24, FaceClassifier& calibrator12, FaceClassifier& calibrator24, vector<Rect>& resultRects, vector<float>& scores);

int GetAllWindows(const Mat& img, vector<Rect>& rects);

class FaceDetector
{
//    struct NetConfig
//    {
//        int     size;
//        string  modelFile;
//        string  trainedFile;
//        float   threshold;
//        float   nmsOverlap;
//    };
//
//    struct NetConfigs
//    {    
//        string              modelFolder;
//        vector<NetConfig>   detectNets;
//        vector<NetConfig>   calibNets;
//        float               gloablNmsOverlap;
//    };
    
public:
    FaceDetector():m_minFaceSize(32), m_scaleStep(1.118f), m_rootSpacing(4),m_rootWindowSize(12)
    {
        GenerateCalibLabelSet();
    }
    FaceDetector(int min_FaceSize, float scaleStep, int spacing):m_minFaceSize(min_FaceSize),m_scaleStep(scaleStep), m_rootSpacing(spacing),m_rootWindowSize(12)
    {
        GenerateCalibLabelSet();
    }
    ~FaceDetector();
    
public:
    void LoadConfigs(const string& protoFilepath);
    void SetDetectors(const NetConfigs& configs);
    void SetCalibrators(const NetConfigs& configs);
    void SetDetectors(const vector<string>& modelFiles, const vector<string>& trainedFiles)
    {
        SetClassifiers(modelFiles, trainedFiles, m_detectors);
    }
    void SetCalibrators(const vector<string>& modelFiles, const vector<string>& trainedFiles)
    {
        SetClassifiers(modelFiles, trainedFiles, m_calibrators);
    }
    
    int Detect(const Mat& img, vector<Rect>& rects, vector<float>& scores);
private:
    void DetectRects(const Mat& img, const FaceClassifier& detector, float threshold, vector<Rect>& rects, vector<float>& scores);
    void CalibrateRects(const Mat& img_ext, const Size& extSize, FaceClassifier* calibrator, FaceClassifier* detector, float threshold_c, float threshold_d, vector<Rect>& rects, vector<float>& scores);

    
    void SetClassifiers(const vector<string>& modelFiles, const vector<string>& trainedFiles, vector<FaceClassifier*>& classifiers);
    int GetSlidingWindows(const Mat& img, vector<Rect>& rects);
    
private:
    vector<FaceClassifier*> m_detectors;    //
    vector<FaceClassifier*> m_calibrators;
    int m_minFaceSize;          // acceptable minimum face size. 40 for FDDB, 80 for AFW.
    float m_scaleStep;          // Detection scale step
    int m_rootSpacing;          // The stride size of sliding window for 12-net
    int m_rootWindowSize;       // Should be 12 for cascaded CNN face detection 
    
    NetConfigs m_configs;              // The settings
};

#endif	/* FACEDETECTOR_H */

