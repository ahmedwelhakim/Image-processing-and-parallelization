#include <iostream>
#include <time.h>
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define THREADS_NO 12 // 12 thread for each channel
#define N 5			  // Kernel size 5x5
using namespace std;
using namespace cv;
struct Arg
{
	Mat src;
	Mat dst;
	double kernel[N][N];
};


const double blurFilter[][N] = {{1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								{1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								{1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								{1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)},
								{1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N), 1.0 / (N * N)}};
const double sharpFilter[][N] = {{0, 0, -1, 0, 0},
								 {0, 0, -1, 0, 0},
								 {-1, -1, 9, -1, -1},
								 {0, 0, -1, 0, 0},
								 {0, 0, -1, 0, 0}};
const double verticalEdgeFilter[][N] = {{0, 0, -1, 0, 0},
										{0, 0, -1, 0, 0},
										{0, 0, 4, 0, 0},
										{0, 0, -1, 0, 0},
										{0, 0, -1, 0, 0}};
const double horizontalEdgeFilter[][N] = {{0, 0, 0, 0, 0},
										  {0, 0, 0, 0, 0},
										  {-1, -1, 4, -1, -1},
										  {0, 0, 0, 0, 0},
										  {0, 0, 0, 0, 0}};



Mat convolute(const Mat src, const double kernel[][N]);
void init_args(struct Arg *arg,Mat b,Mat g, Mat r);
void *threadFunction(void *arg);
void pthread_apply_filters(struct Arg*arg);