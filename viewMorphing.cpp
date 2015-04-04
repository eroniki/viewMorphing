//============================================================================
// Name        : viewMorphing.cpp
// Author      : Murat Ambarkutuk
// Version     :
// Copyright   : Under GPL 2015
// Description : View Morphing, Ansi-style
//============================================================================

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/legacy.hpp"

#include "libViewMorphing.h"

using namespace std;
using namespace cv;

const int alpha_slider_max = 100;
const int beta_slider_max = 100;
int alpha_slider, beta_slider;
double alpha, beta;
char alphaBarText[50];
char betaBarText[50];


void on_trackbar(int, void*){
  alpha = (double) alpha_slider/alpha_slider_max;
  beta = (double) beta_slider/beta_slider_max;
  cout<<"alpha: "<<alpha<<" beta: "<<beta<<endl;
  //imshow("Linear Interpolation", inter);
}

int main(){
	stereoVision cameraX, cameraY;
	morphParameters parameters;
	static bool drawCanvases = true;

	double intrinsicX[] ={534.71330014574596, 0, 335.13862534129095, //intrinsic values
	          	  	  	  0.0, 534.71330014574596, 240.20611211620974, //(320.264)
	          	  	  	  0.0, 0.0, 1.0};

	double intrinsicY[] = {534.71330014574596,	0., 334.01539789486719, //intrinsic values
							0.0, 534.71330014574596, 241.590467217259776, // (394,306)
							0.0, 0.0, 1.0};

	double distX[] = {-0.27456948645081880, -0.018313659520296389, 0., 0., 0., 0., 0., -0.24476896009779484};	//distortion coeff's
	double distY[] = {-0.28073637369061943, 0.093010333969654108, 0., 0., 0., 0., 0., 0.016329629645783102};	//distortion coeff's

	cameraX.intrinsic = Mat(3, 3, CV_64F, intrinsicX);
	cameraX.distortion= Mat(1, 8, CV_64F, distX);
	cameraX.frame = imread("/home/eiki/workspace/viewMorphing/dataset/set-4/left.jpg",1);

	cameraY.intrinsic = Mat(3, 3, CV_64F, intrinsicY);
	cameraY.distortion= Mat(1, 8, CV_64F, distY);
	cameraY.frame = imread("/home/eiki/workspace/viewMorphing/dataset/set-4/right.jpg",1);

	if(!cameraX.frame.data || !cameraY.frame.data){
		cerr<<"Image could loaded!"<<endl;
		return -1;
	}

	// Construct class for morphing
	viewMorphing myMorph(cameraX, cameraY);

	// Initialize Windows showing frames as inputs & results.
	//namedWindow("Camera 0",1);
	//namedWindow("Camera 1",1);
	namedWindow("Camera 0 Undistorted",1);
	namedWindow("Camera 1 Undistorted",1);
	namedWindow("Warped Frame Camera 0",1);
	namedWindow("Warped Frame Camera 1",1);
	namedWindow("Canvas",1);

	// Initialize Track Bars
	sprintf(alphaBarText, "Alpha x %d", alpha_slider_max);
	sprintf(betaBarText, "Beta x %d", beta_slider_max);
	createTrackbar(alphaBarText, "Canvas", &alpha_slider, alpha_slider_max, on_trackbar);
	createTrackbar(betaBarText, "Canvas", &beta_slider, beta_slider_max, on_trackbar);

	while(true){
		// Find features
//		myMorph.featureDetection(cameraX, cameraY, 200);
//		cout<<"Feature Found Camera X: "<<cameraX.keyPointSize<<std::endl;
//		cout<<"Feature Found Camera Y: "<<cameraY.keyPointSize<<std::endl;
//
//		// Extract descriptors
//		myMorph.featureDescriptorExtractor(cameraX, cameraY);
//
//		// Match features
//		myMorph.featureMatcher(cameraX, cameraY,parameters);
//		cout<<"Matches: "<<parameters.matches.size()<<" in which number of: "<<parameters.goodMatches.size()<<" matched good."<<endl;
//
//		// Check if there is enough matching features or not to proceed.
//		if(parameters.goodMatches.size()<40){
//			cerr<<"Couldn't find enough matching key points."<<endl;
//			return -2;
//		}
//
//		cout<<"Matched Key Points Coordinates X:"<<cameraX.matchedKeyPointsCoordinates.size()<<" Y: "<<cameraY.matchedKeyPointsCoordinates.size()<<endl;
//
//		// Use matching features to estimate fundamental matrix (F)
//		myMorph.getFundamentalMatrix(cameraX, cameraY, parameters);
//
//		// Get Essential matrix (E) from F
//		myMorph.getEssentialMatrix(cameraX, cameraY, parameters);
//
//		// Decompose E into Rotation Matrices (R1,R2,R3) and Translation Vector (T)
//		myMorph.decomposeEssentialMatrix(cameraX, cameraY, parameters);
//
//		// Rectify both input images to make them on the same plane.
//		// In^ = In*inverse(Hn)
////		Mat canvasPreWarped;
////		myMorph.preWarp(cameraX, cameraY, canvasPreWarped);
//
//		if(drawCanvases){
//		//	drawMatches(cameraX.frameUndistorted, cameraX.keyPoints, cameraY.frameUndistorted, cameraY.keyPoints, parameters.goodMatches, parameters.canvasKeyPoints, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//		//	imshow("Canvas", parameters.canvasKeyPoints);
////			imshow("canvas",canvasPreWarped);
//		}
//		cout<<parameters.F<<endl;
//		myMorph.uncalibratedRectify(cameraX, cameraY, parameters);
//		imshow("Warped Frame Camera 0", cameraX.preWarped);
//		imshow("Warped Frame Camera 1", cameraY.preWarped);
		imshow("Camera 0 Undistorted", cameraX.frameUndistorted);
		imshow("Camera 1 Undistorted", cameraY.frameUndistorted);
		if(waitKey (30) >= 0) break;

	// TODO Next Steps
	//	myMorph.interpolate();
	//	myMorph.postWarp();
	}

	return 0;
}
