#include"headers.h"

int main()
{
 	struct Arg arg[THREADS_NO];
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

	

	//-----------------------------Clock start------------------------------------------------------------
	clock_t start, end;
	start = clock();
	init_args(arg,b,g,r);
	pthread_apply_filters(arg);

	Mat dst[THREADS_NO / 3];

	for (size_t i = 0; i < THREADS_NO; i += 3)
	{
		vector<Mat> channels;
		channels.push_back(arg[i].dst);
		channels.push_back(arg[i + 1].dst);
		channels.push_back(arg[i + 2].dst);
		dst[i / 3] = src.clone();
		merge(channels, dst[i / 3]);
	}

	end = clock();
	//-----------------------------End of clock------------------------------------------------------------

	double time_taken = (double)(end - start) / (double)(CLOCKS_PER_SEC);
	cout << "Time taken:" << time_taken << endl;
}
void *threadFunction(void *arg)
{

	struct Arg a = *((struct Arg *)arg);
	(*((struct Arg *)arg)).dst = convolute((*((struct Arg *)arg)).src, (*((struct Arg *)arg)).kernel);
	return NULL;
}
void pthread_apply_filters(struct Arg*arg)
{
	pthread_t t[THREADS_NO];

	for (size_t i = 0; i < THREADS_NO; i++)
	{
		pthread_create(&t[i], NULL, threadFunction, ((void *)&arg[i]));
	}
	for (size_t i = 0; i < THREADS_NO; i++)
	{
		pthread_join(t[i], NULL);
	}
	
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
void init_args(struct Arg *arg,Mat b,Mat g, Mat r)
{
	for (size_t t = 0; t < THREADS_NO; t += 3)
	{
		arg[t].src = b.clone();
	}
	for (size_t t = 1; t < THREADS_NO; t += 3)
	{
		arg[t].src = g.clone();
	}
	for (size_t t = 2; t < THREADS_NO; t += 3)
	{
		arg[t].src = r.clone();
	}
	for (size_t c = 0; c < 3; c++)
	{
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				arg[c].kernel[i][j] = blurFilter[i][j];
			}
		}
	}
	for (size_t c = 3; c < 6; c++)
	{
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				arg[c].kernel[i][j] = sharpFilter[i][j];
			}
		}
	}
	for (size_t c = 6; c < 9; c++)
	{
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				arg[c].kernel[i][j] = verticalEdgeFilter[i][j];
			}
		}
	}
	for (size_t c = 9; c < 12; c++)
	{
		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < N; j++)
			{
				arg[c].kernel[i][j] = horizontalEdgeFilter[i][j];
			}
		}
	}
}