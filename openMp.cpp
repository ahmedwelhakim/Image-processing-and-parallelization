#include <iostream>
#include <opencv2/opencv.hpp>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <stdio.h>
#include <omp.h>
using namespace std;
using namespace cv;

#define THREADS_NO 12
#define N 5
const double filters[][N][N] = {{{1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								 {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								 {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								 {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								 {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)}},

			 {{0, 0, -1, 0, 0},
			  {0, 0, -1, 0, 0},
			  {-1, -1, 9, -1, -1},
			  {0, 0, -1, 0, 0},
			  {0, 0, -1, 0, 0}},

			 {{0, 0, -1, 0, 0},
			  {0, 0, -1, 0, 0},
			  {0, 0, 4, 0, 0},
			  {0, 0, -1, 0, 0},
			  {0, 0, -1, 0, 0}},
			 
			 {{0, 0, 0, 0, 0},
			  {0, 0, 0, 0, 0},
			  {-1, -1, 4, -1, -1},
			  {0, 0, 0, 0, 0},
			  {0, 0, 0, 0, 0}}};
			  Mat convolute(const Mat src, const double kernel[][N]);
int main()
{
	Mat src = imread("photo.jpg", IMREAD_COLOR);

	Mat blurImg = src.clone();
	Mat sharpImg = src.clone();
	Mat verticalEdgeImg = src.clone();
	Mat horizontalEdgeImg = src.clone();

	Mat different_Channels[3];		// declaring a matrix with three channels//
	split(src, different_Channels); // splitting images into 3 different channels//
	Mat b = different_Channels[0];	// loading blue channels//
	Mat g = different_Channels[1];	// loading green channels//
	Mat r = different_Channels[2];

	Mat arr[12];
	//-----------------------------Clock start------------------------------------------------------------
	clock_t start, end;
	start = clock();
	omp_set_num_threads(THREADS_NO);
	Mat dst[THREADS_NO / 3];
#pragma omp parallel
	//-------------parallel --------------------------
	{
		int ID = omp_get_thread_num();
		arr[ID] = convolute(different_Channels[ID % 3], filters[(ID / 3)]);
	}
	//-------------parallel --------------------------
	for (size_t i = 0; i < THREADS_NO; i += 3)
	{
		vector<Mat> channels;
		channels.push_back(arr[i]);
		channels.push_back(arr[i+1]);
		channels.push_back(arr[i+2]);
		dst[i / 3] = src.clone();
		merge(channels, dst[i / 3]);
	}
	for (size_t i = 0; i < THREADS_NO/3; i++)
	{
		imshow("d",dst[i]);
			waitKey(0);
	}

	
	end = clock();
	//-----------------------------End of clock------------------------------------------------------------

	double time_taken = (double)(end - start) / (double)(CLOCKS_PER_SEC);
	cout << "Time taken:" << time_taken << endl;
	return 0;
}
Mat convolute(const Mat src, const double kernel[][N])
{
	// Adding padding to the image
	int borderType = BORDER_REPLICATE;
	int top, bottom, left, right;
	top = floor(N / 2);
	bottom = floor(N / 2);
	left = floor(N / 2);
	right = floor(N / 2);
	RNG rng(12345);
	Mat padded_img;
	Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	copyMakeBorder(src, padded_img, top, bottom, left, right, borderType, value);

	Mat dst = padded_img.clone();

	// make convolution using the kernel
	for (int i = N / 2; i < (padded_img).rows - N / 2; i++)
	{
		for (int j = N / 2; j < (padded_img).cols - N / 2; j++)
		{
			int sum = 0;
			for (int k = 0; k < N; k++)
			{
				for (int l = 0; l < N; l++)
				{

					sum += (kernel[k][l] * padded_img.at<uchar>(i - N / 2 + k, j - N / 2 + l));
				}
			}
			if (sum < 0)
				sum = 0;
			else if (sum > 255)
				sum = 255;

			dst.at<uchar>(i, j) = round(sum);
		}
	}

	Mat croped = dst(Range(N / 2, dst.size().height - N / 2), Range(N / 2, dst.size().width - N / 2));

	return croped.clone();
}