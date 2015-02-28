#include "conv_layer.h"
#include "softmax_class.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <ctime>


using namespace std;
using namespace Eigen;
using namespace libcnn;

int main()
{
	/*make up test numbers*/
	MatrixXf image1 = MatrixXf::Random(4,4);
	MatrixXf image2 = MatrixXf::Random(4,4);
	MatrixXf image3 = MatrixXf::Random(4,4);
	MatrixXf image4 = MatrixXf::Random(4,4);
	vector<MatrixXf> img1;
	vector<MatrixXf> img2;
	img1.push_back(image1);
	img1.push_back(image2);
	img2.push_back(image3);
	img2.push_back(image4);
	vector<vector<MatrixXf> > image;
	image.push_back(img1);
	image.push_back(img2);
	
	ConvPoolLayer layer1(image, 1, 3, "same", "tanh", 2, 2);

	for(int i = 0; i < layer1.batch_maps_input.size(); i++)
	{
		for(int j = 0; j < layer1.batch_maps_input[i].size(); j++)
		{
			cout << "The " << j << "-th map of input example " << i << endl;
			cout << layer1.batch_maps_input[i][j] << endl << endl;
		}
	}

	for(int i = 0; i < layer1.weight.size(); i++)
	{
		cout << "The " << i << "-th weight;\n" << layer1.weight[i] << endl;
	}


	for(int i = 0; i < layer1.batch_maps_activated.size(); i++)
	{
		for(int j = 0; j < layer1.batch_maps_activated[i].size(); j++)
		{
			cout << "The " << j << "-th activated map of training example " << i << endl;
			cout << layer1.batch_maps_activated[i][j] << endl << endl;
		}
	}


	MatrixXf err1(2,2);
	MatrixXf err2(2,2);
	MatrixXf err3(2,2);
	MatrixXf err4(2,2);	
	err1 << 1,2,3,4;
	err2 << 5,6,7,8;
	err3 << 9,10,11,12;
	err4 << 13,14,15,16;
	vector<MatrixXf> error1;
	vector<MatrixXf> error2;
	error1.push_back(err1);
	error1.push_back(err2);
	error2.push_back(err3);
	error2.push_back(err4);
	vector<vector<MatrixXf> > error;
	error.push_back(error1);
	error.push_back(error2);


	vector<vector<MatrixXf> > gradient_conved;
//	layer1.bp_pool_activate(error, gradient_conved);
//	layer1.bp_conv(gradient_conved, 0.1);
	layer1.back_prop(error, 0.1);
//	layer1.bp_pool_activate(error, grads);
/*
	MatrixXf input1(6,6);
	input1 << 0,0,0,0,0,0,0,0,0,1,1,0,0,2,2,3,3,0,0,4,4,5,5,0,0,6,6,7,7,0,0,0,0,0,0,0;
	MatrixXf input2(6,6);
	input2 << 0,0,0,0,0,0,0,9,8,7,6,0,0,5,4,3,2,0,0,2,3,4,5,0,0,6,7,8,9,0,0,0,0,0,0,0;	
	vector<MatrixXf> input;
	input.push_back(input1);
	input.push_back(input2);

	MatrixXf error1(4,4);
	MatrixXf error2(4,4);
	MatrixXf error3(4,4);
	MatrixXf error4(4,4);
	error1 << 0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0;
	error2 << 1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0;
	error3 << 0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0;
	error4 << 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1;
	vector<MatrixXf> error;
	error.push_back(error1);
	error.push_back(error2);
	error.push_back(error3);
	error.push_back(error4);
	
	MatrixXf weight1(3,3);
	MatrixXf weight2(3,3);
	weight1 << 1,0,0,0,0,1,0,1,0;
	weight2 << 0,1,0,1,0,0,0,0,1;
	vector<MatrixXf> weight;
	weight.push_back(weight1);
	weight.push_back(weight2);

	vector<MatrixXf> grad_weight;
	MatrixXf grad_bias;
	vector<MatrixXf> grad_input;
*/
	

	return 0;
}
