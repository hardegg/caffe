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
int FaceDetection(const Mat& img, FaceClassifier& detector, FaceClassifier& calibrator, vector<Rect>& resultRects, vector<float>& scores);
int GetAllWindows(const Mat& img, vector<Rect>& rects);

#endif	/* FACEDETECTOR_H */

