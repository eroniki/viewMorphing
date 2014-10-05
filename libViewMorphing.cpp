/*
 * libViewMorphing.cpp
 *
 *  Created on: Jul 31, 2014
 *      Author: eiki
 */

#include "libViewMorphing.h"

// private functions
bool viewMorphing::isInFrontOfBothCameras(std::vector<cv::Point3d> inlierX, std::vector<cv::Point3d> inlierY, cv::Mat R, cv::Mat T){
	for(unsigned int i=0;i<inlierX.size();i++){
		cv::Mat pX(inlierX.at(i));
		cv::Mat pY(inlierY.at(i));
		cv::Mat num = (R.row(0)-pY.at<double>(0,0)*R.row(2))*T;
		cv::Mat denum = (R.row(0)-pY.at<double>(0,0)*R.row(2))*pY;
		cv::Mat ratio = (num/denum);

		double firstZ = ratio.at<double>(0,0);
		cv::Point3d pX3D = cv::Point3d((pX.at<double>(0,0)*firstZ),(double)(pY.at<double>(0,0)*firstZ),firstZ);

		cv::Mat pY3Dp = (R.t()*cv::Mat(pX3D)) - (R.t()*T);
		cv::Point3d pY3D(pY3Dp.at<double>(0,0),pY3Dp.at<double>(0,1),pY3Dp.at<double>(0,2));

		if(pX3D.y<0 || pY3D.y<0)
			return false;
	}

	return true;
}

// public functions
viewMorphing::viewMorphing(){
	std::cout<<"Constructing View Morphing.."<<std::endl;
}

viewMorphing::viewMorphing(stereoVision& _X, stereoVision& _Y){
	std::cout<<"Constructing View Morphing.."<<std::endl;
	std::cout<<"Initializing Camera 0"<<std::endl;
	_X.whichCamera = 0;
	_X.frameSize = _X.frame.size();
	_X.inverseIntrinsic = _X.intrinsic.inv();

	std::cout<<"Initializing Camera 1"<<std::endl;
	_Y.whichCamera = 1;
	_Y.frameSize = _Y.frame.size();
	_Y.inverseIntrinsic = _Y.intrinsic.inv();

	// Undistort frames to eliminate barrel distortion
	cv::undistort(_X.frame, _X.frameUndistorted, _X.intrinsic, _X.distortion);
	cv::undistort(_Y.frame, _Y.frameUndistorted, _Y.intrinsic, _Y.distortion);

	// Produce Gray Images from Undistorted RGB ones for speed concerns.
	cv::cvtColor(_X.frameUndistorted,_X.frameGray,CV_RGB2GRAY);
	cv::cvtColor(_Y.frameUndistorted,_X.frameGray,CV_RGB2GRAY);
	// Produce 3 channel Gray Images
	cv::cvtColor(_X.frameGray,_X.frameGray3Channels,CV_GRAY2RGB);
	cv::cvtColor(_Y.frameGray,_Y.frameGray3Channels,CV_GRAY2RGB);
	std::cout<<"Construction is OK! Moving on..."<<std::endl;
}

viewMorphing::~viewMorphing(){
	std::cout<<"Deconstructor of cameraCalibration Class.."<<std::endl;
}

void viewMorphing::featureDetection(stereoVision& _X, stereoVision& _Y, int _minHessian){
	cv::SurfFeatureDetector _detector(_minHessian);
	_detector.detect(_X.frameUndistorted, _X.keyPoints);
	_detector.detect(_Y.frameUndistorted, _Y.keyPoints);

	_X.keyPointSize = _X.keyPoints.size();
	_Y.keyPointSize = _Y.keyPoints.size();
}

void viewMorphing::featureDescriptorExtractor(stereoVision& _X, stereoVision& _Y){
	cv::SurfDescriptorExtractor _extractor;

	_extractor.compute(_X.frameUndistorted, _X.keyPoints, _X.descriptors);
	_extractor.compute(_Y.frameUndistorted, _Y.keyPoints, _Y.descriptors);
}

void viewMorphing::featureMatcher(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters, double _minDist, double _maxDist){
	cv::FlannBasedMatcher _matcher;
	_parameters.matches.clear();
	_parameters.goodMatches.clear();
	_X.matchedKeyPointsCoordinates.clear();
	_Y.matchedKeyPointsCoordinates.clear();
	_matcher.match(_X.descriptors, _Y.descriptors, _parameters.matches);
	// TODO Make a proper threshold for this variable, try.. except..

	if(_parameters.matches.size()<100){
		std::cerr<<"Not Enough matches between key points."<<std::endl;
	}

	for(int i = 0; i < _X.descriptors.rows; i++){
		double dist = _parameters.matches[i].distance;
	    if(dist < _minDist) _minDist = dist;
	    if(dist > _maxDist) _maxDist = dist;
	}

	for(int i = 0; i < _X.descriptors.rows; i++){
		//TODO A heuristic approach needed for setting this scale. Change algorithm into recursive one.
		if(_parameters.matches[i].distance < 4*_minDist){
			_parameters.goodMatches.push_back(_parameters.matches[i]);
		}
	}

	for(int i = 0; i < (int)_parameters.goodMatches.size(); i++){
		_X.matchedKeyPointsCoordinates.push_back(_X.keyPoints[_parameters.goodMatches[i].queryIdx].pt);
		_Y.matchedKeyPointsCoordinates.push_back(_Y.keyPoints[_parameters.goodMatches[i].trainIdx].pt);
	}
}

void viewMorphing::getFundamentalMatrix(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters){
	_parameters.F = cv::findFundamentalMat(cv::Mat(_X.matchedKeyPointsCoordinates), cv::Mat(_Y.matchedKeyPointsCoordinates), CV_FM_RANSAC, 3.0, 0.99, mask);
}

void viewMorphing::getEssentialMatrix(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters){
	_parameters.E = _X.intrinsic.t()*_parameters.F*_X.intrinsic;
}

void viewMorphing::decomposeEssentialMatrix(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters){
	cv::SVD svd(_parameters.E,cv::SVD::MODIFY_A);
	cv::Mat svd_u = svd.u;
	cv::Mat svd_vt = svd.vt;
	cv::Mat svd_w = svd.w;
	cv::Matx33d W(0,-1,0,1,0,0,0,0,1);//HZ 9.13
	cv::Mat_<double> R = svd_u * cv::Mat(W) * svd_vt; //HZ 9.19
	cv::Mat_<double> t = svd_u.col(2); //u3

	// TODO Revise here. Is it really needed?
//	P1 = cv::Matx34f(R(0,0),R(0,1),R(0,2),t(0),
//	             R(1,0),R(1,1),R(1,2),t(1),
//	             R(2,0),R(2,1),R(2,2),t(2));

	std::vector<cv::Point3d> inlierX;
	std::vector<cv::Point3d> inlierY;
	for(unsigned int i=0; i<mask.total(); i++){
		if(mask.at<unsigned char>(i,1)){
			cv::Matx31d X(_X.matchedKeyPointsCoordinates[i].x,_X.matchedKeyPointsCoordinates[i].y,1.0);
			cv::Matx31d Y(_Y.matchedKeyPointsCoordinates[i].x,_Y.matchedKeyPointsCoordinates[i].y,1.0);
			cv::Mat mulX = _X.inverseIntrinsic*(cv::Mat(X));
			cv::Mat mulY = _Y.inverseIntrinsic*(cv::Mat(Y));
			cv::Point3d pX(mulX.at<double>(0,0),mulX.at<double>(1,0),mulX.at<double>(2,0));
			cv::Point3d pY(mulY.at<double>(0,0),mulY.at<double>(1,0),mulY.at<double>(2,0));
			inlierX.push_back(pX);
			inlierY.push_back(pY);
		}
	}
	_X.R = cv::Mat(R).clone();
	_X.t = cv::Mat(svd_u.col(2)).clone();

	if(!isInFrontOfBothCameras(inlierX,inlierY,_X.R,_X.t)){
		std::cout<<"birinci degil t degisti"<<std::endl;
		_X.t = -1*_X.t;
		if(!isInFrontOfBothCameras(inlierX,inlierY,_X.R,_X.t)){
			std::cout<<"ikinci degil R ve t degisti"<<std::endl;
			_X.R = svd_u * cv::Mat(W).t() * svd_vt;
			_X.t = -1*_X.t;
			if(!isInFrontOfBothCameras(inlierX,inlierY,_X.R,_X.t)){
				std::cout<<"ucuncu de degilmis, t degisti"<<std::endl;
				_X.t = -1*_X.t;
			}
		}
	}
}

void viewMorphing::preWarp(stereoVision& _X, stereoVision& _Y, cv::Mat& _canvas){
	cv::Mat R1, R2, P1, P2, Q, mapx1, mapx2, mapy1, mapy2;
	cv::Rect validROI[2];

	double sf;
	int w, h;
	sf = 600./MAX(_X.frameSize.width, _X.frameSize.height);
	w = cvRound(_X.frameSize.width*sf);
	h = cvRound(_X.frameSize.height*sf);
	_canvas.create(h, w*2, CV_8UC3);

	cv::stereoRectify(_X.intrinsic, _X.distortion, _Y.intrinsic, _Y.distortion, _X.frameSize, _X.R, _X.t, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, _X.frameSize, &validROI[0], &validROI[1]);

	cv::initUndistortRectifyMap(_X.intrinsic, _X.distortion, R1, P1, _X.frameSize, CV_32FC1, mapx1, mapy1);
	cv::initUndistortRectifyMap(_Y.intrinsic, _Y.distortion, R2, P2, _Y.frameSize, CV_32FC1, mapx2, mapy2);

	cv::remap(_X.frameGray, _X.preWarped, mapx1, mapy1, CV_INTER_LINEAR);
	cv::remap(_Y.frameGray, _Y.preWarped, mapx2, mapy2, CV_INTER_LINEAR);

	//for(int k=0; k<2;k++){
		int k = 0;
		cv::Mat canvasPart = _canvas(cv::Rect(w*k, 0, w, h));
		cv::resize(_X.preWarped, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
		std::cout<<"k:"<<k<<" X: "<<validROI[k].x<<" Y: "<<validROI[k].y<<" size: "<<validROI[k].size()<<std::endl;
	    cv::Rect vroi(cvRound(validROI[k].x*sf), cvRound(validROI[k].y*sf), cvRound(validROI[k].width*sf), cvRound(validROI[k].height*sf));
		cv::rectangle(canvasPart, vroi, cv::Scalar(0,0,255), 3, 8);
		k = 1;
		cv::Mat canvasPart2 = _canvas(cv::Rect(w*k, 0, w, h));
		cv::resize(_Y.preWarped, canvasPart2, canvasPart2.size(), 0, 0, CV_INTER_AREA);
		std::cout<<"k:"<<k<<" X: "<<validROI[k].x<<" Y: "<<validROI[k].y<<" size: "<<validROI[k].size()<<std::endl;
	    cv::Rect vroi2(cvRound(validROI[k].x*sf), cvRound(validROI[k].y*sf), cvRound(validROI[k].width*sf), cvRound(validROI[k].height*sf));
		cv::rectangle(canvasPart2, vroi2, cv::Scalar(0,0,255), 3, 8);
	// }

}

void viewMorphing::interpolate(){

}

void viewMorphing::postWarp(){

}

void viewMorphing::uncalibratedRect(stereoVision& _X, stereoVision& _Y, morphParameters& _parameters){
	cv::Mat H1,H2,preWrappedLeft,preWrappedRight;
	cv::Mat R1,R2,P1,P2,mapx1,mapx2,mapy1,mapy2;
	//std::cout<<matchedKeyPointCoordinatesX.size()<<"  "<<matchedKeyPointCoordinatesY.size()<<std::endl;
	cv::stereoRectifyUncalibrated(_X.matchedKeyPointsCoordinates, _Y.matchedKeyPointsCoordinates, _parameters.F, _X.frameSize, H1, H2, 0.8);
    R1 = _X.intrinsic.inv()*H1*_X.intrinsic;
    R2 = _Y.intrinsic.inv()*H2*_Y.intrinsic;
    P1 = _X.intrinsic;
    P2 = _Y.intrinsic;

    cv::initUndistortRectifyMap(_X.intrinsic, _X.distortion, R1, P1, _X.frameSize, CV_16SC2, mapx1, mapy1);
    cv::initUndistortRectifyMap(_Y.intrinsic, _Y.distortion, R2, P2, _Y.frameSize, CV_16SC2, mapx2, mapy2);

    cv::remap(_X.frameUndistorted, _X.preWarped, mapx1, mapy1, CV_INTER_LINEAR);
    cv::remap(_Y.frameUndistorted, _Y.preWarped, mapx2, mapy2, CV_INTER_LINEAR);
}
