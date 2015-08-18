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

using namespace cv;

void GenerateCalibLabelSet();
Rect CalibrateRect(const Rect rect, float ex, float ey, float es);
int FaceDetection(const Mat& img, FaceClassifier& detector12, FaceClassifier& detector24, FaceClassifier& calibrator12, FaceClassifier& calibrator24, vector<Rect>& resultRects, vector<float>& scores);

int GetAllWindows(const Mat& img, vector<Rect>& rects);

class FaceDetector
{
public:
    FaceDetector():m_minFaceSize(32),m_rootSpacing(4),m_rootWindowSize(12)
    {
        GenerateCalibLabelSet();
    }
    FaceDetector(int min_FaceSize, int spacing):m_minFaceSize(min_FaceSize),m_rootSpacing(spacing),m_rootWindowSize(12)
    {
        GenerateCalibLabelSet();
    }
public:
    void SetDetectors(const vector<string>& modelFiles, const vector<string>& trainedFiles)
    {
        SetClassifiers(modelFiles, trainedFiles, m_detectors);

    }
    void SetCalibrators(const vector<string>& modelFiles, const vector<string>& trainedFiles)
    {
        SetClassifiers(modelFiles, trainedFiles, m_calibrators);

    }
    
    void Detect(const Mat& img, vector<Rect>& rects, vector<float>& scores);
private:
    void SetClassifiers(const vector<string>& modelFiles, const vector<string>& trainedFiles, vector<FaceClassifier*>& classifiers);
    int GetSlidingWindows(const Mat& img, vector<Rect>& rects);
private:
    vector<FaceClassifier*> m_detectors;    //
    vector<FaceClassifier*> m_calibrators;
    int m_minFaceSize;          // acceptable minimum face size. 40 for FDDB, 80 for AFW.
    int m_rootSpacing;          // The stride size of sliding window for 12-net
    int m_rootWindowSize;       // Should be 12 for cascaded CNN face detection
};

#endif	/* FACEDETECTOR_H */

