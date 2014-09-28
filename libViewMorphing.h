/*
 * libViewMorphing.h
 *
 *  Created on: Jul 31, 2014
 *      Author: eiki
 */

#include <iostream>
#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/legacy.hpp"

#ifndef LIBVIEWMORPHING_H_
#define LIBVIEWMORPHING_H_

class viewMorphing{
// XXX Add Private Variables & Functions
	bool verbose;
	cv::Mat intrinsicX, intrinsicY, distortionCoeffs; // Intrinsic Matrixes
	cv::Mat descriptorsX, descriptorsY;
	cv::Mat intrinsicXInverse, intrinsicYInverse;
	cv::Mat frameGrayX3C,frameGrayY3C, frameMatches;
	std::vector<cv::Point2f> matchedKeyPointCoordinatesX, matchedKeyPointCoordinatesY;
	std::vector<cv::KeyPoint> keypointsX, keypointsY;
	std::vector<cv::DMatch> matches, good_matches;
	std::vector<cv::Point3f> linesX, linesY;
	cv::RNG rng;
	cv::FlannBasedMatcher matcher;
	bool isInFrontOfBothCameras(std::vector<cv::Point3d> inlierX, std::vector<cv::Point3d> inlierY, cv::Mat R, cv::Mat T);
public:
	// XXX Add Public Variables & Functions
	cv::Mat warpedFrameX, warpedFrameY;
	cv::Mat frameX, frameY;
	cv::Mat frameGrayX, frameGrayY;
	cv::Mat frameXUndistorted, frameYUndistorted;
	cv::Mat E, F; // Essential and Fundamental Matrix
	cv::Matx34f P1; // Projection Matrix
	cv::Mat Rot, T; // Rotation and translation
	viewMorphing();
	viewMorphing(cv::Mat intX, cv::Mat intY, cv::Mat distortionCoeff, bool isVerbose = false);
	~viewMorphing();
	void displayFrames();
	void featureDetection(cv::Mat frameX, cv::Mat frameY, int minHessian=400);
	void featureDescriptorExtractor(cv::Mat frameX, cv::Mat frameY);
	int featureMatcher(double maxDist=0, double minDist=100, bool draw = false);
	void getFundamentalMatrix();
	void getEssentialMatrix();
	void decomposeEssentialMatrix();
	void initMorph(double scale=1.0);
	void preWarp();
	void interpolate();
	void postWarp();
	void uncalibratedRect();
};

#endif /* LIBVIEWMORPHING_H_ */
