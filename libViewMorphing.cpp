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
	//cv::namedWindow("Frame X",1);
	//cv::namedWindow("Frame Y",1);
	cv::namedWindow("Warped Frame X",1);
	cv::namedWindow("Warped Frame Y",1);
	//cv::namedWindow("Frame X Gray",1);
	//cv::namedWindow("Frame Y Gray",1);
	cv::namedWindow("Frame X Undistorted",1);
	cv::namedWindow("Frame Y Undistorted",1);
	rng(12345);
	verbose = false;
	std::cout<<"Constructing View Morphing.."<<std::endl;
}

viewMorphing::viewMorphing(cv::Mat intX, cv::Mat intY, cv::Mat distortionCoeff, bool isVerbose){
	//cv::namedWindow("Frame X",1);
	//cv::namedWindow("Frame Y",1);
	cv::namedWindow("Warped Frame X",1);
	cv::namedWindow("Warped Frame Y",1);
	//cv::namedWindow("Frame X Gray",1);
	//cv::namedWindow("Frame Y Gray",1);
	cv::namedWindow("Frame X Undistorted",1);
	cv::namedWindow("Frame Y Undistorted",1);
	rng(12345);
	verbose = isVerbose;
	intrinsicX = intX.clone();
	intrinsicY = intY.clone();
	distortionCoeffs = distortionCoeff.clone();
	intrinsicXInverse = intrinsicX.inv();
	intrinsicYInverse = intrinsicY.inv();
	std::cout<<"Constructing View Morphing.."<<std::endl;
}

viewMorphing::viewMorphing(stereoVision X, stereoVision Y, bool isVerbose){
	//cv::namedWindow("Frame X",1);
	//cv::namedWindow("Frame Y",1);
	cv::namedWindow("Warped Frame X",1);
	cv::namedWindow("Warped Frame Y",1);
	//cv::namedWindow("Frame X Gray",1);
	//cv::namedWindow("Frame Y Gray",1);
	cv::namedWindow("Frame X Undistorted",1);
	cv::namedWindow("Frame Y Undistorted",1);
	rng(12345);
	verbose = isVerbose;
	std::cout<<"Constructing View Morphing.."<<std::endl;
}

viewMorphing::~viewMorphing(){
	std::cout<<"Deconstructor of cameraCalibration Class.."<<std::endl;
}

void viewMorphing::displayFrames(){
	//cv::imshow("Frame X",frameX);
	//cv::imshow("Frame Y",frameY);
	//cv::imshow("Frame X Gray", frameGrayX);
	//cv::imshow("Frame Y Gray", frameGrayY);
	cv::imshow("Frame X Undistorted", frameXUndistorted);
	cv::imshow("Frame Y Undistorted", frameYUndistorted);
	cv::waitKey(0);
// TODO Show warped frames too.
//cv::imshow("Warped Frame X", warpedFrameX);
//cv::imshow("Warped Frame Y", warpedFrameY);
}

std::vector<unsigned int> viewMorphing::featureDetection(cv::Mat frameX, cv::Mat frameY, int minHessian){
	cv::SurfFeatureDetector detector(minHessian);
	detector.detect(frameX, keypointsX);
	detector.detect(frameY, keypointsY);
	unsigned int size[2]= {keypointsX.size(),keypointsY.size()};
	std::vector<unsigned int> sizes(size, size + sizeof(size)/sizeof(unsigned int));
	return sizes;
}

void viewMorphing::featureDescriptorExtractor(cv::Mat frameX, cv::Mat frameY){
	cv::SurfDescriptorExtractor extractor;

	extractor.compute(frameX, keypointsX, descriptorsX);
	extractor.compute(frameY, keypointsY, descriptorsY);
}

int viewMorphing::featureMatcher(double maxDist, double minDist, bool draw){
	matcher.match(descriptorsX, descriptorsY, matches);
	// TODO Make a proper threshold for this variable, try.. except..
	std::cout<<"Matches: "<<matches.size()<<std::endl;
	if(matches.size()<100){
		std::cerr<<"Not Enough matches between key points."<<std::endl;
		return -1;
	}

	for(int i = 0; i < descriptorsX.rows; i++){
		double dist = matches[i].distance;
	    if(dist < minDist) minDist = dist;
	    if(dist > maxDist) maxDist = dist;
	}

	for(int i = 0; i < descriptorsX.rows; i++){
		//TODO A heuristic approach needed for setting this scale. Change algorithm into recursive one.
		if(matches[i].distance < 4*minDist){
			good_matches.push_back(matches[i]);
		}
	}
	if(draw){
		drawMatches(frameXUndistorted, keypointsX, frameYUndistorted, keypointsY, good_matches, frameMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imshow("Matching Key Points",frameMatches);
	}
	for(int i = 0; i < (int)good_matches.size(); i++){
		matchedKeyPointCoordinatesX.push_back(keypointsX[good_matches[i].queryIdx].pt);
		matchedKeyPointCoordinatesY.push_back(keypointsX[good_matches[i].trainIdx].pt);
	}

	return good_matches.size();
}

void viewMorphing::getFundamentalMatrix(){
	F = findFundamentalMat(matchedKeyPointCoordinatesX,matchedKeyPointCoordinatesY,mask);
	std::cout<<"F: "<<F<<std::endl;
}

void viewMorphing::getEssentialMatrix(){
	E = intrinsicX.t()*F*intrinsicX;
	std::cout<<"E: "<<E<<std::endl;
}

void viewMorphing::decomposeEssentialMatrix(){
	cv::SVD svd(E,cv::SVD::MODIFY_A);
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
			cv::Matx31d X(matchedKeyPointCoordinatesX[i].x,matchedKeyPointCoordinatesX[i].y,1.0);
			cv::Matx31d Y(matchedKeyPointCoordinatesY[i].x,matchedKeyPointCoordinatesY[i].y,1.0);
			cv::Mat mulX = intrinsicXInverse*(cv::Mat(X));
			cv::Mat mulY = intrinsicYInverse*(cv::Mat(Y));
			cv::Point3d pX(mulX.at<double>(0,0),mulX.at<double>(1,0),mulX.at<double>(2,0));
			cv::Point3d pY(mulY.at<double>(0,0),mulY.at<double>(1,0),mulY.at<double>(2,0));
			inlierX.push_back(pX);
			inlierY.push_back(pY);
		}
	}
	Rot = cv::Mat(R).clone();
	T = cv::Mat(svd_u.col(2)).clone();
	if(!isInFrontOfBothCameras(inlierX,inlierY,Rot,T)){
		std::cout<<"birinci degil t degisti"<<std::endl;
		T = -1*T;
		if(!isInFrontOfBothCameras(inlierX,inlierY,Rot,T)){
			std::cout<<"ikinci degil R ve t degisti"<<std::endl;
			Rot = svd_u * cv::Mat(W).t() * svd_vt;
			T = -1*T;
			if(!isInFrontOfBothCameras(inlierX,inlierY,Rot,T)){
				std::cout<<"ucuncu de degilmis, t degisti"<<std::endl;
				T = -1*T;
			}
		}
	}
}

void viewMorphing::initMorph(double scale){
	cv::undistort(frameX, frameXUndistorted, intrinsicX, distortionCoeffs);
	cv::undistort(frameY, frameYUndistorted, intrinsicY, distortionCoeffs);

	if(scale != 1.0){
		cv::resize(frameXUndistorted,frameXUndistorted,cv::Size(0,0),scale,scale);
		cv::resize(frameYUndistorted,frameYUndistorted,cv::Size(0,0),scale,scale);
	}
	// Produce Gray Images from Undistorted RGB ones for speed concerns.
	cv::cvtColor(frameXUndistorted,frameGrayX,CV_RGB2GRAY);
	cv::cvtColor(frameYUndistorted,frameGrayY,CV_RGB2GRAY);
	// Produce 3 channel Gray Images
	cv::cvtColor(frameGrayX,frameGrayX3C,CV_GRAY2RGB);
	cv::cvtColor(frameGrayY,frameGrayY3C,CV_GRAY2RGB);
}

void viewMorphing::preWarp(){
	cv::Mat R1, R2, P1, P2, Q, mapx1, mapx2, mapy1, mapy2;
	cv::Mat rectX, rectY;
	cv::Mat canvas;
	cv::Rect validROI[2];

	double sf;
	int w, h;
	sf = 600./MAX(frameX.size().width, frameX.size().height);
	w = cvRound(frameX.size().width*sf);
	h = cvRound(frameX.size().height*sf);
	canvas.create(h, w*2, CV_8UC3);

	cv::stereoRectify(intrinsicX, distortionCoeffs, intrinsicY, distortionCoeffs, frameX.size(), Rot, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, frameX.size(), &validROI[0], &validROI[1]);

	cv::initUndistortRectifyMap(intrinsicX, distortionCoeffs, R1, P1, frameX.size(), CV_32FC1, mapx1, mapy1);
	cv::initUndistortRectifyMap(intrinsicY, distortionCoeffs, R2, P2, frameX.size(), CV_32FC1, mapx2, mapy2);
	cv::remap(frameXUndistorted, rectX, mapx1, mapy1, CV_INTER_LINEAR);
	cv::remap(frameYUndistorted, rectY, mapx2, mapy2, CV_INTER_LINEAR);

	//for(int k=0; k<2;k++){
		int k = 0;
		cv::Mat canvasPart = canvas(cv::Rect(w*k, 0, w, h));
		cv::resize(rectX, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
		std::cout<<"k:"<<k<<" X: "<<validROI[k].x<<" Y: "<<validROI[k].y<<" size: "<<validROI[k].size()<<std::endl;
	    cv::Rect vroi(cvRound(validROI[k].x*sf), cvRound(validROI[k].y*sf), cvRound(validROI[k].width*sf), cvRound(validROI[k].height*sf));
		cv::rectangle(canvasPart, vroi, cv::Scalar(0,0,255), 3, 8);
		k = 1;
		cv::Mat canvasPart2 = canvas(cv::Rect(w*k, 0, w, h));
		cv::resize(rectY, canvasPart2, canvasPart2.size(), 0, 0, CV_INTER_AREA);
		std::cout<<"k:"<<k<<" X: "<<validROI[k].x<<" Y: "<<validROI[k].y<<" size: "<<validROI[k].size()<<std::endl;
	    cv::Rect vroi2(cvRound(validROI[k].x*sf), cvRound(validROI[k].y*sf), cvRound(validROI[k].width*sf), cvRound(validROI[k].height*sf));
		cv::rectangle(canvasPart2, vroi2, cv::Scalar(0,0,255), 3, 8);
	// }

	cv::imshow("canvas",canvas);
	cv::imshow("Warped Frame X", rectX);
	cv::imshow("Warped Frame Y", rectY);
}

void viewMorphing::interpolate(){

}

void viewMorphing::postWarp(){

}

void viewMorphing::uncalibratedRect(){
	cv::Mat H1,H2,preWrappedLeft,preWrappedRight;
	cv::Mat R1,R2,P1,P2,mapx1,mapx2,mapy1,mapy2;
	//std::cout<<matchedKeyPointCoordinatesX.size()<<"  "<<matchedKeyPointCoordinatesY.size()<<std::endl;
	cv::stereoRectifyUncalibrated(matchedKeyPointCoordinatesX, matchedKeyPointCoordinatesY, F, frameX.size(), H1, H2);
    R1 = intrinsicX.inv()*H1*intrinsicX;
    R2 = intrinsicY.inv()*H2*intrinsicY;
    P1 = intrinsicX;
    P2 = intrinsicY;
    cv::initUndistortRectifyMap(intrinsicX, distortionCoeffs, R1, P1, frameX.size(), CV_16SC2, mapx1, mapy1);
    cv::initUndistortRectifyMap(intrinsicY, distortionCoeffs, R2, P2, frameX.size(), CV_16SC2, mapx2, mapy2);

    cv::remap(frameXUndistorted, preWrappedLeft, mapx1, mapy1, CV_INTER_LINEAR);
    cv::remap(frameYUndistorted, preWrappedRight, mapx2, mapy2, CV_INTER_LINEAR);

	cv::imshow("Uncalibrated Rectified Image Left", preWrappedLeft);
	cv::imshow("Uncalibrated Rectified Image Right", preWrappedRight);
}
