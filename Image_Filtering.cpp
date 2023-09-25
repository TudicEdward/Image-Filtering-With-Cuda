#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Filtering_Functions.h"

using namespace std;
using namespace cv;

int main() {
	Mat Image = imread("Example.png");
	Mat Image2 = imread("Example.png");
	Mat Image3 = imread("Example.png");
	Mat Image4 = imread("Example.png");
	Mat Image5 = imread("Example.png");
	Mat Image6 = imread("Example.png");

	cout << "The uploaded image has Height: " << Image.rows << ", Width: " << Image.cols << endl;

	Image_Negative(Image.data, Image.rows, Image.cols);
	imwrite("Negative_Filter.png", Image);

	
	Image_Grayscale(Image2.data, Image2.rows, Image2.cols);
	imwrite("Grayscale_Filter.png", Image2);

	
	Image_Sepia(Image3.data, Image3.rows, Image3.cols);
	imwrite("Sepia_Filter.png", Image3);


	Image_Red(Image4.data, Image4.rows, Image4.cols);
	imwrite("Red_Filter.png", Image4);


	Image_Green(Image5.data, Image5.rows, Image5.cols);
	imwrite("Green_Filter.png", Image5);


	Image_Blue(Image6.data, Image6.rows, Image6.cols);
	imwrite("Blue_Filter.png", Image6);

	system("pause");
	return 0;
}