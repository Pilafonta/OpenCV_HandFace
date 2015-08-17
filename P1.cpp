/**
	CS440_P1.cpp
	@author: Peter LaFontaine
	@version: V1
    @date: 2/13/15

	CS585 Image and Video Computing Fall 2014
	Assignment 2
	--------------
	This program:
		 Recognizes hand shapes or gestures and creates a graphical display 
	--------------
	
*/

#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat frame;
Mat frame0;
Mat frameDest;
RNG rng(12345);

//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
    int m = a;
    (void)((m < b) && (m = b));
    (void)((m < c) && (m = c));
    return m;
}

//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
    int m = a;
    (void)((m > b) && (m = b));
    (void)((m > c) && (m = c));
    return m;
}

//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
    //Surveys of skin color modeling and detection techniques:
    //Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    //Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            //For each pixel, compute the average intensity of the 3 color channels
            Vec3b intensity = src.at<Vec3b>(i,j); //Vec3b is a vector of 3 uchar (unsigned character)
            int B = intensity[0]; int G = intensity[1]; int R = intensity[2];
            if ((R > 40 && G > 40 && B > 20) && (myMax(R,G,B) - myMin(R,G,B) > 15) && (abs(R-G) > 15) && (R > G) && (R > B)){
                dst.at<uchar>(i,j) = 255;
            }
        }
    }
}

//Finds the length between points
float pointLength(Point a, Point b){
    float d= sqrt(fabs( pow(a.x-b.x,2) + pow(a.y-b.y,2) )) ;
    return d;
}

//Finds the angle between two lengths
float getAngle(Point s, Point f, Point e){
    float l1 = pointLength(f,s);
    float l2 = pointLength(f,e);
    float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
    float angle = acos(dot/(l1*l2));
    angle=angle*180/(3.14);
    return angle;
}


int main()
{
	VideoCapture cap(0); 

	// if not successful, exit program
    if (!cap.isOpened())  
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
	
	
	namedWindow("MyVideo",WINDOW_AUTOSIZE);
    
	while (1)
    {
        

		// read a new frame from video
        bool bSuccess = cap.read(frame); 

		//if not successful, break loop
        if (!bSuccess) 
        {
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }
        frameDest = Mat::zeros(frame.rows, frame.cols, CV_8UC3); //Returns a zero array of same size as src mat
        frameDest = frame.clone();
        
        
		if (waitKey(30) == 27) 
		{
			cout << "esc key is pressed by user" << endl;
			break; 
		}
        Mat tmp = Mat::zeros(frameDest.size(), CV_8UC1);
        
        //detects the whether there is skin in the camera.
        mySkinDetect(frameDest, tmp);
        
        //Different vector and matrix initializations
        vector<vector<cv::Point> > countours;
        vector<Vec4i> hierarchy;
        
        findContours(tmp, countours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
        Mat contOut = Mat::zeros(tmp.size(),CV_8UC1);
        vector<vector<Point> >hull( countours.size() );
        vector<vector<Point> > defect_points(countours.size());
        vector<vector<Vec4i> >convexOut( countours.size() );
        vector<vector<int> > hullI(countours.size());

        int count = 0;

        Mat drawing = Mat::zeros( tmp.size(), CV_8UC3 );
        for(int i = 0; i<countours.size(); i++){
            
            int area = contourArea(countours[i]);
            
            if(area>30000){
                //Gets the outlines of the figures from the camera
                drawContours(contOut, countours, i, 255, CV_FILLED, 8, hierarchy);
                convexHull(Mat(countours[i]), hull[i],false);
                convexHull(Mat(countours[i]), hullI[i], false);
                if (hullI[i].size()>3){
                    convexityDefects(Mat(countours[i]), hullI[i], convexOut[i]);
                }
                
                //Finds the points and the lengths between the points
                for(int k = 0; k<convexOut[i].size(); k++){
                    if(convexOut[i][k][3]>20*256)
                    {
                        //initalizes the points using convexOut
                        int ind_0=convexOut[i][k][0];
                        int ind_1=convexOut[i][k][1];
                        int ind_2=convexOut[i][k][2];
                        
                        //Actually draws the points and lines on the screen
                        defect_points[i].push_back(countours[i][ind_2]);
                        cv::circle(drawing,countours[i][ind_0],5,Scalar(0,255,0),-1);
                        cv::circle(drawing,countours[i][ind_1],5,Scalar(0,255,0),-1);
                        cv::circle(drawing,countours[i][ind_2],5,Scalar(0,0,255),-1);
                        cv::line(drawing,countours[i][ind_2],countours[i][ind_0],Scalar(0,0,255),1);
                        cv::line(drawing,countours[i][ind_2],countours[i][ind_1],Scalar(0,0,255),1);
                        
                        //cleans up the screen
                        erode(drawing,drawing,3);
                        //calculates the angles between the points
                        float angle = getAngle(countours[i][ind_0], countours[i][ind_1], countours[i][ind_2]);
                        if(angle>15 && angle<100){
                            count = count +1;
                        }
                    }
                }
                //draws the contours of the shapes on the screen
                drawContours( drawing, countours, i, 255, 1, 8, vector<Vec4i>(), 0, Point() );
                drawContours( drawing, hull, i, 255, 1, 8, vector<Vec4i>(), 0, Point() );
            }
        }
        
        //distinguishes whether the object is a face or a hand.
        if (count < 5 && count != 0){
            cv::putText(drawing, "Face", cvPoint(30,30),FONT_HERSHEY_COMPLEX, 0.8, cvScalar(200,200,250), 1, CV_AA);

        }
        if (count >5){
            cv::putText(drawing, "Hand", cvPoint(30,30),FONT_HERSHEY_COMPLEX, 0.8, cvScalar(200,200,250), 1, CV_AA);

        }
        
        //shows the screen
        imshow("Countours", drawing);
        

	}
    
	
	cap.release();
	return 0;
}

