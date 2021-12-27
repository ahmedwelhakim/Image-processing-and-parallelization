#include <iostream>
#include <time.h>
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace std;
using namespace cv;

#define THREADS_NO 4
#define N 5 //Kernel size 5x5

struct Arg
{
	Mat src;
	Mat dst;
	double kernel[N][N];
};

Mat convolute(const Mat src, double kernel[][N]);
void* threadFunction(void* arg)
{

	struct Arg a = *((struct Arg*)arg);
	(*((struct Arg*)arg)).dst = convolute(a.src, a.kernel);
	return NULL;

}
int main()
{
	struct Arg arg[THREADS_NO];
	Mat src = imread("photo.jpg", IMREAD_COLOR);

	Mat blurImg = src.clone();
	Mat sharpImg = src.clone();
	Mat verticalEdgeImg = src.clone();
	Mat horizontalEdgeImg = src.clone();

	double blurFilter[][N] = { {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
							  {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
							  {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
							  {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
							  {1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)} };
	double sharpFilter[][N] = { {0, 0, -1, 0, 0},
							   {0, 0, -1, 0, 0},
							   {-1, -1, 9, -1, -1},
							   {0, 0, -1, 0, 0},
							   {0, 0, -1, 0, 0} };
	double verticalEdgeFilter[][N] = { {0, 0, -1, 0, 0},
									  {0, 0, -1, 0, 0},
									  {0, 0, 4, 0, 0},
									  {0, 0, -1, 0, 0},
									  {0, 0, -1, 0, 0} };
	double horizontalEdgeFilter[][N] = { {0, 0, 0, 0, 0},
										{0, 0, 0, 0, 0},
										{-1, -1, 4, -1, -1},
										{0, 0, 0, 0, 0},
										{0, 0, 0, 0, 0} };

	for (size_t t = 0; t < THREADS_NO; t++)
	{
		arg[t].src = src.clone();
	}
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			arg[0].kernel[i][j] = blurFilter[i][j];
		}
	}
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			arg[1].kernel[i][j] = sharpFilter[i][j];
		}
	}
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			arg[2].kernel[i][j] = verticalEdgeFilter[i][j];
		}
	}
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			arg[3].kernel[i][j] = horizontalEdgeFilter[i][j];
		}
	}
	//-----------------------------Clock start------------------------------------------------------------
	clock_t start, end;
	start = clock();

	pthread_t t[THREADS_NO];

	for (size_t i = 0; i < THREADS_NO; i++)
	{
		pthread_create(&t[i], NULL, threadFunction, ((void*)&arg[i]));
	}
	for (size_t i = 0; i < THREADS_NO; i++)
	{
		pthread_join(t[i], NULL);
	}

	end = clock();
	//-----------------------------End of clock------------------------------------------------------------
	
	double time_taken = (double)(end - start) / (double)(CLOCKS_PER_SEC);
	cout << "Time taken:" << time_taken << endl;
	
}

Mat convolute(const Mat src, double kernel[][N])
{
	// Adding padding to the image
	int borderType = BORDER_REPLICATE;
	int top, bottom, left, right;
	top = (int)(N / 2);
	bottom = (int)(N / 2);
	left = (int)(N / 2);
	right = (int)(N / 2);
	RNG rng(12345);
	Mat padded_img;
	Scalar value(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	copyMakeBorder(src, padded_img, top, bottom, left, right, borderType, value);

	Mat different_Channels[3];			   //declaring a matrix with three channels//
	split(padded_img, different_Channels); //splitting images into 3 different channels//
	Mat b = different_Channels[0];		   //loading blue channels//
	Mat g = different_Channels[1];		   //loading green channels//
	Mat r = different_Channels[2];		   //loading red channels//

	Mat b_out = b.clone();
	Mat g_out = g.clone();
	Mat r_out = r.clone();
	// make convolution using the kernel
	for (int i = N / 2; i < (padded_img).rows - N / 2; i++)
	{
		for (int j = N / 2; j < (padded_img).cols - N / 2; j++)
		{
			int sum_b = 0;
			int sum_g = 0;
			int sum_r = 0;
			for (int k = 0; k < N; k++)
			{
				for (int l = 0; l < N; l++)
				{

					sum_b += (kernel[k][l] * b.at<uchar>(i - N / 2 + k, j - N / 2 + l));
					sum_g += (kernel[k][l] * g.at<uchar>(i - N / 2 + k, j - N / 2 + l));
					sum_r += (kernel[k][l] * r.at<uchar>(i - N / 2 + k, j - N / 2 + l));
				}
			}
			if (sum_b < 0)
				sum_b = 0;
			else if (sum_b > 255)
				sum_b = 255;

			if (sum_g < 0)
				sum_g = 0;
			else if (sum_g > 255)
				sum_g = 255;

			if (sum_r < 0)
				sum_r = 0;
			else if (sum_r > 255)
				sum_r = 255;

			b_out.at<uchar>(i, j) = round(sum_b);
			g_out.at<uchar>(i, j) = round(sum_g);
			r_out.at<uchar>(i, j) = round(sum_r);
		}
	}
	vector<Mat> channels;
	channels.push_back(b_out);
	channels.push_back(g_out);
	channels.push_back(r_out);
	Mat dst = src.clone();
	merge(channels, dst);

	Mat croped = dst(Range(N / 2, dst.size().height - N / 2), Range(N / 2, dst.size().width - N / 2));

	return croped.clone();
}