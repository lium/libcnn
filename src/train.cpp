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
#include "yaml-cpp/yaml.h"

//#define _FILE
//#define DEBUG
//#define CHANNEL
//#define LABEL
//#define SENSE
//#define VLR
//#define PARAM
//#define STP
#define MAIN
//#define POST
//#define SAVE

using namespace std;
using namespace Eigen;
using namespace libcnn;
using std::cout;

const int image_height = 320;
const int image_width = 240;
const int pool = 8;
const int num_kerns1 = 4;
const int num_kerns2 = 6;
const int num_kerns3 = 8;
const int kernel_size = 7;
const int num_labels = 9;
const double weight_bound = 0.1;
const double bias_bound = 0.01;
const double wb = 0.05;
const double bb = 0.0005;
const int num_input_channels = 3;
const int pool_size = 2;
double learning_rate1 = 0.025;
double learning_rate2 = 0.025;
const int num_images = 20;
const int batch_size = 5;
const int num_batches = num_images / batch_size;
int beginning = 400;
const int loop = 300;

void print_conv(const vector<MatrixXd>& weights, const MatrixXd& biases)
{
	for(int i = 0; i < weights.size(); i++)
	{
		cout << "convolutional weight:\n" << weights[i] << endl;	
	}
	cout << "bias:\n" << biases.transpose() << endl;
}

void print_class(const MatrixXd& weight, const MatrixXd& bias)
{
	cout << "classifier weight:\n" << weight << endl;
	cout << "classifier bias:\n" << bias << endl;
}

void training_single()
{
	/* load in image files and label files */
	vector<vector<MatrixXd> > all_images;                                         			// total images
	VectorXi all_labels; 														  			// total labels
	data_generation(image_height, image_width, beginning, num_images, all_images, all_labels);	  			// loading

	/* unit scaling of images and sampling of labels*/
	unit_scaling(all_images);													  			// scale the input to range 1
	VectorXi samp_labels(num_images*40*30);														// sampled useful labels
	label_sampling(image_height, image_width, pool, num_images, all_labels, samp_labels);   // label sampling operator
	
	/* randomly initialize parameters */
	vector<int> num_kernels;
	num_kernels.push_back(num_kerns1); 	num_kernels.push_back(num_kerns2); 	num_kernels.push_back(num_kerns3); 
	vector<vector<MatrixXd> > weights, weights_backup;	
	vector<MatrixXd> biases, biases_backup;			
	rand_conv(weights, biases, num_kernels, kernel_size, num_labels, weight_bound, bias_bound);
	weights_backup = weights;
	biases_backup = biases;
	MatrixXd class_weight, class_weight_backup;
	MatrixXd class_bias, class_bias_backup;
	rand_classifier(class_weight, class_bias, num_kernels, num_labels, num_input_channels, wb, bb);
	class_weight_backup = class_weight;
	class_bias_backup = class_bias;
	
	/* select few examples to test */
	vector<vector<MatrixXd> > image1;
	copy(all_images.begin(), all_images.begin() + 1, back_inserter(image1));
	VectorXi label1;
//	label1 = samp_labels.segment(1200*beginning, 1200);	
	label1 = samp_labels.segment(0, 1200);	
	#ifdef MAIN
	/* construct the structure for training */
	for(int counter = 0; counter < loop; counter++)
	{

		ConvPoolLayer layer_conv0(image1, num_kerns1, kernel_size, "same", "tanh", weights[0], biases[0]);
		Pool layer_pool0(layer_conv0.batch_maps_activated, pool_size, pool_size);
		ConvPoolLayer layer_conv1(layer_pool0.output_batch_pooled, num_kerns2, kernel_size, "same", "tanh", weights[1], biases[1]);
		Pool layer_pool1(layer_conv1.batch_maps_activated, pool_size, pool_size);
		ConvPoolLayer layer_conv2(layer_pool1.output_batch_pooled, num_kerns2, kernel_size, "same", "tanh", weights[2], biases[2]);
		Pool layer_pool2(layer_conv2.batch_maps_activated, pool_size, pool_size);

	
		MatrixXd feature_vectors(9, 1200);
		get_feature_vector(layer_pool2.output_batch_pooled, feature_vectors);
		Softmax classifier(feature_vectors, num_labels, label1, weight_bound, bias_bound, class_weight, class_bias);
		classifier.calculation_output();
		
//		cout << "cost @ " << counter << ": " << classifier.cost << endl;
		cout << classifier.cost << endl;
		double cost = classifier.cost;	
//		vec_cost.push_back(cost);


		classifier.backprop(learning_rate1);
		vector<vector<MatrixXd> > grad_s2c;
		int map_height = 40;
		int map_width = 30;
		softmax2conv(classifier.grad_input, batch_size, map_height, map_width, grad_s2c);
		layer_pool2.back_prop(grad_s2c);
		layer_conv2.back_prop(layer_pool2.grad_batch_input, learning_rate2);
		layer_pool1.back_prop(layer_conv2.grad_batch_maps_input);
		layer_conv1.back_prop(layer_pool1.grad_batch_input, learning_rate2);
		layer_pool0.back_prop(layer_conv1.grad_batch_maps_input);
		layer_conv0.back_prop(layer_pool0.grad_batch_input, learning_rate2);
		
		weights[0] = layer_conv0.weight; biases[0] = layer_conv0.bias;
		weights[1] = layer_conv1.weight; biases[1] = layer_conv1.bias;
		weights[2] = layer_conv2.weight; biases[2] = layer_conv2.bias;	
		class_weight = classifier.weight; class_bias = classifier.bias;

	/* postprocessing */
/*		if(counter == loop - 1)
		{
			int accuracy = 0;
			for(int i = 0; i < label1.size(); i++)		
			{
				if(classifier.m[i] == label1[i])
				{
					accuracy++;
				}
			}	
			cout << "correct predictions: " << accuracy << endl;
			cout << classifier.m.transpose() << endl;
		}
*/
	}	
	save(weights, biases, class_weight, class_bias);
	#endif

}

void training_batch()
{
	/* load in image files and label files */
	vector<vector<MatrixXd> > all_images;                                         			// total images
	VectorXi all_labels; 														  			// total labels
	data_generation(image_height, image_width, beginning, num_images, all_images, all_labels);	  			// loading

	/* unit scaling of images and sampling of labels*/
	unit_scaling(all_images);													  			// scale the input to range 1
	VectorXi samp_labels(num_images*40*30);														// sampled useful labels
	label_sampling(image_height, image_width, pool, num_images, all_labels, samp_labels);   // label sampling operator

	/* randomly initialize parameters */
	vector<int> num_kernels;
	num_kernels.push_back(num_kerns1); 	num_kernels.push_back(num_kerns2); 	num_kernels.push_back(num_kerns3); 
	vector<vector<MatrixXd> > weights;	
	vector<MatrixXd> biases;			
	rand_conv(weights, biases, num_kernels, kernel_size, num_labels, weight_bound, bias_bound);
	MatrixXd class_weight;
	MatrixXd class_bias;
	rand_classifier(class_weight, class_bias, num_kernels, num_labels, num_input_channels, wb, bb);
//	print_params(weights, biases, class_weight, class_bias);

	ConvPoolLayer* layer_conv0; ConvPoolLayer* layer_conv1; ConvPoolLayer* layer_conv2;
	Pool* layer_pool0; Pool* layer_pool1; Pool* layer_pool2;
	Softmax* classifier;	
	
	#ifdef MAIN
	/* construct the structure for training */
	for(int counter = 0; counter < loop; counter++)
	{
//		if((counter+1) % 50 == 0)
//		{
//			learning_rate1 += 0.05;
//			learning_rate2 += 0.05;	
//		}
		/* select training batch*/
		srand(time(NULL));
		beginning = (rand() % num_batches) * batch_size;
//		beginning = (rand() % 1) * batch_size;
		vector<vector<MatrixXd> > image1;
		copy(all_images.begin() + beginning, all_images.begin() + beginning + batch_size, back_inserter(image1));
		VectorXi label1;
		label1 = samp_labels.segment(1200*beginning, 1200*batch_size);	
		/* training */
		layer_conv0 = new ConvPoolLayer(image1, num_kerns1, kernel_size, "same", "tanh", weights[0], biases[0]);
		layer_pool0 = new Pool(layer_conv0->batch_maps_activated, pool_size, pool_size);
		layer_conv1 = new ConvPoolLayer(layer_pool0->output_batch_pooled, num_kerns2, kernel_size, "same", "tanh", weights[1], biases[1]);
		layer_pool1 = new Pool(layer_conv1->batch_maps_activated, pool_size, pool_size);
		layer_conv2 = new ConvPoolLayer(layer_pool1->output_batch_pooled, num_kerns2, kernel_size, "same", "tanh", weights[2], biases[2]);
		layer_pool2 = new Pool(layer_conv2->batch_maps_activated, pool_size, pool_size);
	
		MatrixXd feature_vectors(9, 1200);
		get_feature_vector(layer_pool2->output_batch_pooled, feature_vectors);
		classifier = new Softmax(feature_vectors, num_labels, label1, weight_bound, bias_bound, class_weight, class_bias);
		
		classifier->calculation_output();
		cerr << "selected training batch: " << (beginning / batch_size);	
		cerr << " cost @ " << counter << ": " << classifier->cost << endl;

		classifier->backprop(learning_rate1);
		vector<vector<MatrixXd> > grad_s2c;
		int map_height = 40;
		int map_width = 30;
		softmax2conv(classifier->grad_input, batch_size, map_height, map_width, grad_s2c);
		layer_pool2->back_prop(grad_s2c);
		layer_conv2->back_prop(layer_pool2->grad_batch_input, learning_rate2);
		layer_pool1->back_prop(layer_conv2->grad_batch_maps_input);
		layer_conv1->back_prop(layer_pool1->grad_batch_input, learning_rate2);
		layer_pool0->back_prop(layer_conv1->grad_batch_maps_input);
		layer_conv0->back_prop(layer_pool0->grad_batch_input, learning_rate2);
		weights[0] = layer_conv0->weight; biases[0] = layer_conv0->bias;
		weights[1] = layer_conv1->weight; biases[1] = layer_conv1->bias;
		weights[2] = layer_conv2->weight; biases[2] = layer_conv2->bias;	
		class_weight = classifier->weight; class_bias = classifier->bias;

		delete(layer_conv0); delete(layer_conv1); delete(layer_conv2);
		delete(layer_pool0); delete(layer_pool1); delete(layer_pool2);
		delete(classifier);
	/* postprocessing */
		#ifdef POST
		if(counter == loop - 1)
		{
			int accuracy = 0;
			for(int i = 0; i < label1.size(); i++)		
			{
				if(classifier->m[i] == label1[i])
				{
					accuracy++;
				}
			}	
			cout << "correct predictions: " << accuracy << endl;
			cout << classifier->m.transpose() << endl;
		}
		#endif
	}	
	#endif
	save(weights, biases, class_weight, class_bias);
}

void test_load()
{
	vector<vector<MatrixXd> > weights;
	vector<MatrixXd> biases;
	MatrixXd class_weight;
	MatrixXd class_bias;
	vector<int> kerns;
	kerns.push_back(4);
	kerns.push_back(6);
	kerns.push_back(8);
	load(kerns, weights, biases, class_weight, class_bias);
	for(int i = 0; i < weights.size(); i++)
	{
		for(int j = 0; j < weights[i].size(); j++)
		{
			cout << "weight" << i << j << endl;
			cout << weights[i][j] << endl;
		}
		cout << "bias" << i << endl;
		cout << biases[i] << endl; 
	}
	cout << class_weight << endl;
	cout << class_bias << endl;
}

int main()
{
	training_single();
//	training_batch();
//	test_load();
	return 0;
}
