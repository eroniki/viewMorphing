/*
 * libViewMorphing.h
 *
 *  Created on: Jul 31, 2014
 *      Author: Murat Ambarkutuk
 */

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#ifndef LIBVIEWMORPHING_H_
#define LIBVIEWMORPHING_H_

struct stereoVision{
	  unsigned int whichCamera;
	  unsigned int keyPointSize;
	  unsigned int featureSize;
	  unsigned int descriptorSize;
	  cv::Mat intrinsic;
	  cv::Mat inverseIntrinsic;
	  cv::Mat distortion;
	  cv::Mat projection;
	  cv::Mat F, E;
	  cv::Mat R;
	  cv::Mat t;
	  // frames
	  cv::Mat frame;
	  cv::Mat frameGray;
	  cv::Mat frameGray3Channels;
	  cv::Mat frameUndistorted;
	  cv::Mat preWarped;
	  // Sizes of frames
	  cv::Size frameSize;
	  // key points, feature, descriptors
	  std::vector<cv::Point2f> matchedKeyPointsCoordinates;
	  std::vector<cv::KeyPoint> keyPoints;
	  cv::Mat descriptors;
	  std::vector<cv::Point3f> lines;
};

struct morphParameters{
	cv::Mat F, E;
	std::vector<cv::DMatch> matches, goodMatches;
	cv::Mat canvasKeyPoints;
};

class viewMorphing{
// XXX Add Private Variables & Functions
	cv::Mat mask; // mask for validate fundamental matrix
	std::vector<cv::Point3f> linesX, linesY;
	bool isInFrontOfBothCameras(std::vector<cv::Point3d> inlierX, std::vector<cv::Point3d> inlierY, cv::Mat R, cv::Mat T);
public:
	// XXX Add Public Variables & Functions
	viewMorphing();
	viewMorphing(stereoVision& _X, stereoVision& _Y);
	~viewMorphing();
	void featureDetection(stereoVision& _X, stereoVision& _Y, int _minHessian=400);
	void featureDescriptorExtractor(stereoVision& _X, stereoVision& _Y);
	void featureMatcher(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters, double _minDist=100, double _maxDist=0);
	void getFundamentalMatrix(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters);
	void getEssentialMatrix(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters);
	void decomposeEssentialMatrix(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters);
	void preWarp(stereoVision& _X, stereoVision& _Y, cv::Mat& _canvas);
	void interpolate();
	void postWarp();
	void uncalibratedRectify(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters);
};

#endif /* LIBVIEWMORPHING_H_ */
