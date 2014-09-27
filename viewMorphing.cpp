//============================================================================
// Name        : viewMorphing.cpp
// Author      : Murat Ambarkutuk
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

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
#include "libViewMorphing.h"

using namespace std;
using namespace cv;

const int alpha_slider_max = 100;
const int beta_slider_max = 100;
int alpha_slider, beta_slider;
double alpha, beta;

double intrinsicX[] ={1918.270000, 2.489820, -17.915, //intrinsic values
          	  	  	  0.0, 1922.580000, 63.736, //(320.264)
          	  	  	  0.0, 0.0, 1.0};

double intrinsicY[] = {1909.910000,	0.571503, 33.069000, //intrinsic values
						0.0, 1915.890000, -10.306, // (394,306)
						0.0, 0.0, 1.0};

double dist[] = {-0.0, 0.0, -0.0, 0.0, 0.00000};	//distortion coeff's

char alphaBarText[50];
char betaBarText[50];

// FIXME Delete the unnecessary one's.
Mat warpedLeftFrame,warpedRightFrame, inter; // Pre-warped Images and Linear Interpolated Image
Mat H1,H2, Hs,E,F;			// Sparse Matrix of Projection M.

Mat R1, R2, P1_, P2_, Q;	 // R's rotation matrixes, P's translation matrixes, Q
Mat cameraMatrixLeft, rotMatrixLeft, cameraMatrixRight, rotMatrixRight, transMatrixLeft, transMatrixRight, mask;
Matx34d P;
Matx34d P1;
bool verbose = false;

void on_trackbar(int, void*){
  alpha = (double) alpha_slider/alpha_slider_max;
  beta = (double) beta_slider/beta_slider_max;
  cout<<"alpha: "<<alpha<<" beta: "<<beta<<endl;
  //imshow("Linear Interpolation", inter);
}

int main(){
	Mat distortion(1, 5, CV_64F, dist);	 // distortion coeff's
	Mat intrinsicMatrixX(3, 3, CV_64F, intrinsicX); // intrinsic matrix
	Mat intrinsicMatrixY(3, 3, CV_64F, intrinsicY); // intrinsic matrix

	viewMorphing myMorph(intrinsicMatrixX,intrinsicMatrixY, distortion);

	myMorph.frameX = imread("/home/eiki/workspace/viewMorphing/dataset/set-3/MSR3DVideo-Ballet/cam0/color-cam0-f000.jpg",1);
	myMorph.frameY = imread("/home/eiki/workspace/viewMorphing/dataset/set-3/MSR3DVideo-Ballet/cam3/color-cam3-f000.jpg",1);

	if(!myMorph.frameX.data || !myMorph.frameY.data){
		cout<<"Image could loaded"<<endl;
		return -1;
	}

// TODO Uncomment to view trackbars
// 	view maximum translations with respect to the each axis
//	namedWindow("Linear Interpolation",1);
//	sprintf(alphaBarText, "Alpha x %d", alpha_slider_max);
//	sprintf(betaBarText, "Beta x %d", beta_slider_max);
//	createTrackbar(alphaBarText, "Linear Interpolation", &alpha_slider, alpha_slider_max, on_trackbar);
//	createTrackbar(betaBarText, "Linear Interpolation", &beta_slider, beta_slider_max, on_trackbar);

	myMorph.initMorph();

	myMorph.featureDetection(myMorph.frameXUndistorted, myMorph.frameYUndistorted);
	myMorph.featureDescriptorExtractor(myMorph.frameXUndistorted, myMorph.frameYUndistorted);
// TODO Make a proper threshold for this variable, try.. except..
	int goodMatches = myMorph.featureMatcher(0,100,true);
	if(goodMatches<10)
		return -2;
	myMorph.getFundamentalMatrix();
	myMorph.getEssentialMatrix();
	myMorph.decomposeEssentialMatrix();
	myMorph.preWarp();
	myMorph.uncalibratedRect();

// TODO Do it step by step
//	myMorph.interpolate();
//	myMorph.postWarp();

	myMorph.displayFrames();
	return 0;
}
