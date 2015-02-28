#include "conv_layer.h"
#include "softmax_class.h"
#include "pool.h"
#include "load_data.h"
#include "init_params.h"
#include "preprocess.h"
#include "rand_initialize.h"
#include "read_write.h"
#include "save_load.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <ctime>
#include <sys/types.h>
#include <dirent.h>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <iterator>

using namespace std;
using namespace Eigen;
using namespace libcnn;

const int num_kerns0 = 4;
const int num_kerns1 = 6;
const int num_kerns2 = 8;
const int height = 320;
const int width = 240;
int beginning = 474;
const int num_images = 1;
const int pool = 8;
const int kernel_size = 7;
const int pool_size = 2;
const int num_labels = 9;
const double wb = 1;
const double bb = 1;

void upsample(vector<vector<MatrixXd> >& vec_vec_mat)
{
	for(std::vector<vector<MatrixXd> >::iterator it1 = vec_vec_mat.begin(); it1 != vec_vec_mat.end(); it1++)
	{
		for(std::vector<MatrixXd>::iterator it2 = (*it1).begin(); it2 != (*it1).end(); it2++)
		{
			cv::Mat cv_mat;
			cv::Mat cv_resized_mat;
			eigen2cv((*it2), cv_mat);
			cv::Size size(height, width);
			cv::resize(cv_mat, cv_resized_mat, size);
			cv2eigen(cv_resized_mat, (*it2));
		}
	}
}

void validate()
{
	/* load parameters */
	vector<vector<MatrixXd> > weights;
	vector<MatrixXd> biases;
	MatrixXd class_weight;
	MatrixXd class_bias;
	vector<int> num_kerns;
	num_kerns.push_back(num_kerns0); num_kerns.push_back(num_kerns1); num_kerns.push_back(num_kerns2);
	load(num_kerns, weights, biases, class_weight, class_bias); 

	/* load validating set */
	vector<vector<MatrixXd> > validate_images;
	VectorXi validate_labels;
	data_generation(height, width, beginning, num_images, validate_images, validate_labels);
	/* preprocess of data */
	unit_scaling(validate_images);

	VectorXi samp_labels(num_images* 30 * 40);
	label_sampling(height, width, pool, num_images, validate_labels, samp_labels);

	/* forward propagation */
	ConvPoolLayer* layer_conv0; ConvPoolLayer* layer_conv1; ConvPoolLayer* layer_conv2;
	Pool* layer_pool0; Pool* layer_pool1; Pool* layer_pool2;
	Softmax* classifier;

	layer_conv0 = new ConvPoolLayer(validate_images, num_kerns0, kernel_size, "same", "tanh", weights[0], biases[0]);
	layer_pool0 = new Pool(layer_conv0->batch_maps_activated, pool_size, pool_size);
	layer_conv1 = new ConvPoolLayer(layer_pool0->output_batch_pooled, num_kerns1, kernel_size, "same", "tanh", weights[1], biases[1]);
	layer_pool1 = new Pool(layer_conv1->batch_maps_activated, pool_size, pool_size);
	layer_conv2 = new ConvPoolLayer(layer_pool1->output_batch_pooled, num_kerns2, kernel_size, "same", "tanh", weights[2], biases[2]);
	layer_pool2 = new Pool(layer_conv2->batch_maps_activated, pool_size, pool_size);

	upsample(layer_pool2->output_batch_pooled);
	MatrixXd feature_vectors;
	get_feature_vector(layer_pool2->output_batch_pooled, feature_vectors);
	classifier = new Softmax(feature_vectors, num_labels, validate_labels, wb, bb, class_weight, class_bias);
//	classifier = new Softmax(feature_vectors, num_labels, samp_labels, wb, bb, class_weight, class_bias);

	classifier->calculation_output();
	cout << "predicted_values = [ \n";
	cout << classifier->m.transpose() << "];" <<  endl;
	cout << "accurate_values = [ \n";
	cout << validate_labels.transpose() << "];"<< endl;
//	cout << samp_labels.transpose() << "]"<< endl;
}

int main()
{
	validate();
	return 0;
}
