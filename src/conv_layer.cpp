#include "conv_layer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <string>
#include <stdexcept>
#include <sys/time.h>
#include <omp.h>

//#define CONV_ACT
//#define POOL
//#define GET_FV
//#define SFT_CONV
//#define CONVPOOL
//#define BP
//#define BP_ACT


using namespace std;
using namespace Eigen;
using namespace libcnn;

namespace libcnn{

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::conv2d
 *  Description: This function is used for 2d-image-convolution, in 'valid' mode. Strictly this is autocorrelation rather than convolution
 *    Arguments: map_input: input image to be convolved
 *				    kernel: convolution kernel
 *				map_conved: convolved map
 * =====================================================================================
 */
void ConvPoolLayer::conv2d(const MatrixXd& map_input, const MatrixXd& kernel, MatrixXd& map_conved)
{
	map_conved.setZero(map_input.rows() + 1 - kernel.rows(), map_input.cols() + 1 - kernel.cols()); // initialization of a convolved map with border cut
    int i=0, j=0;
    #pragma omp parallel for schedule(dynamic,1) collapse(2) private(i,j)
	for(i = 0; i < map_conved.rows(); i++)
	{
		for(j = 0; j < map_conved.cols(); j++)
		{
			map_conved(i,j) = (map_input.block(i, j, kernel.rows(), kernel.cols()).array() * kernel.array()).sum(); // strictly this operation is autocorrelation 
		}
	}
}		/* -----  end of function ConvPoolLayer::conv2d  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::conv2d
 *  Description: This function is used for multiple input 2d-image-convolution with multiple kernels, in 'valid' mode
 *               i.e. to colvolve each input image with each kernel
 *    Arguments: maps_input: input images to be convolved
 *				    kernels: convolution kernels
 *				maps_conved: convolved maps
 * =====================================================================================
 */
void ConvPoolLayer::conv2d(const vector<MatrixXd>& maps_input, const vector<MatrixXd> kernels, vector<MatrixXd>& maps_conved)
{
	maps_conved.clear();
	maps_conved.reserve(kernels.size() * maps_input.size());  // number of output maps is equal to the number of input maps times the number of kernels
	MatrixXd map_conved_temp;                               // map_conved_temp is a temporary convolved map to store one single convolution operation

	for(int i = 0; i < kernels.size(); i++)
	{
		for(int j = 0; j < maps_input.size(); j++)
		{
			conv2d(maps_input[i], kernels[j], map_conved_temp);
			maps_conved.push_back(map_conved_temp);			
		}	
	}
}		/* -----  end of function ConvPoolLayer::conv2d  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::flip_conv2d
 *  Description: This function is used for single input 2d-image-convolution with a single kernels, in 'valid' mode. Strictly this is convolution
 *    Arguments: map_input: input image to be convolved
 *				    kernel: convolution kernel
 *				map_conved: convolved map
 * =====================================================================================
 */
void ConvPoolLayer::flip_conv2d(const MatrixXd& map_input, const MatrixXd& kernel, MatrixXd& map_conved)
{
	map_conved.resize(map_input.rows() + 1 - kernel.rows(), map_input.cols() + 1 - kernel.cols());
	conv2d(map_input, kernel.reverse(), map_conved);
}		/* -----  end of function ConvPoolLayer::flip_conv2d  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::activate
 *  Description: This function is used to perform the activation operator after convolution for a single map
 *         TODO: modify this function such that it is in the form of a*tanh(bx), choose a and b to be optimal
 *    Arguments:    map_conved: input image to be activated
 *				          bias: bias term to be added before activation
 *				 map_activated: activated map
 * =====================================================================================
 */
void ConvPoolLayer::activate(const MatrixXd& map_conved, const double& bias, MatrixXd& map_activated)
{
	map_activated.resize(map_conved.rows(), map_conved.cols());
    int i = 0, j = 0;
    #pragma omp parallel for schedule(dynamic,1) collapse(2) private(i,j)
	for(i = 0; i < map_conved.rows(); i++)
	{
		for(j = 0; j < map_conved.cols(); j++)
		{
			map_activated(i,j) = tanh(map_conved(i,j) + bias);
		}	
	}
}		/* -----  end of function ConvPoolLayer::activate  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::activate
 *  Description: This function is used to perform the activation operator after convolution for multiple maps
 *         TODO: check where this function is used, and make sure the bias vector is correctly added
 *    Arguments:    maps_conved: input images to be activated
 *				      vec_ bias: bias term to be added before activation
 *				 maps_activated: activated maps
 * =====================================================================================
 */
void ConvPoolLayer::activate(const vector<MatrixXd>& maps_conved, const MatrixXd& vec_bias, vector<MatrixXd>& maps_activated)
{
	maps_activated.clear();
	maps_activated.reserve(maps_conved.size());
	MatrixXd map_activated_temp;
	for(int i = 0; i < maps_conved.size(); i++)
	{
		activate(maps_conved[i], vec_bias(i,1), map_activated_temp);
		maps_activated.push_back(map_activated_temp);
	}
}		/* -----  end of function ConvPoolLayer::activate  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::conv2d_activate
 *  Description: This function is used to perform convolution and activation together for a single input
 *    Arguments:     map_input: input images to be operated
 *                      kernel: convolution kernel
 *				          bias: bias term to be added before activation
 *				 map_activated: activated map
 * =====================================================================================
 */
void ConvPoolLayer::conv2d_activate(const MatrixXd& map_input, const MatrixXd& kernel, const double& bias, const string& type, MatrixXd& map_activated)
{
	map_activated.resize(map_input.rows() + 1 - kernel.rows(), map_input.cols() + 1 - kernel.cols());
	double value_conved;
    int i = 0, j = 0;
    #pragma omp parallel for schedule(dynamic,1) collapse(2) private(i,j,value_conved)
	for(i = 0; i < map_activated.rows(); i++)
	{
		for(j = 0; j < map_activated.cols(); j++)
		{	
			value_conved = (map_input.block(i, j, kernel.rows(), kernel.cols()).array() * kernel.array()).sum(); 
//			#ifdef CONV_ACT
//			cout << "convolved value: " << value_conved << endl;
//			#endif
			if(type == "tanh")
			{
				map_activated(i,j) = tanh(value_conved + bias);
			}
			else if(type == "sigmoid")
			{
				map_activated(i,j) = sigmoid(value_conved + bias);
			}
		}
	}
	#ifdef CONV_ACT
	cout << "input:\n" << map_input << endl;
	cout << "kernel:\n" << kernel << endl;
	cout << "bias:\n" << bias << endl;
	cout << "output:\n" << map_activated << endl;
	#endif
}		/* -----  end of function ConvPoolLayer::conv2d_activate  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::conv2d_activate
 *  Description: This function is used to perform convolution and activation together for a multiple inputs
 *    Arguments:     maps_input: input images to be operated
 *                       weight: convolution kernel
 *				       bias_vec: bias term to be added before activation
 *				 maps_activated: activated maps
 * =====================================================================================
 */
void ConvPoolLayer::conv2d_activate(const vector<MatrixXd>& maps_input, const vector<MatrixXd>& weight, const MatrixXd& bias_vec, const string& type, vector<MatrixXd>& maps_activated)
{
	maps_activated.clear();
	maps_activated.reserve(weight.size() * maps_input.size());
	MatrixXd map_activated;                                    // temporary matrix to store activated map

    int i = 0, j = 0;
	for(i = 0; i < weight.size(); i++)
	{
		for(j = 0; j < maps_input.size(); j++)
		{
            conv2d_activate(maps_input[j], weight[i], bias_vec(i,0), type, map_activated);
            maps_activated.push_back(map_activated);
		}
	}	
}		/* -----  end of function ConvPoolLayer::conv2d_activate  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::get_feature_vector
 *  Description: This function is used to convert a set of feature maps into corresponding feature vectors
 *    Arguments: batch_feature_maps: batch of trained feature maps
 *                  feature_vectors: output feature vectors    
 * =====================================================================================
 */
void get_feature_vector(const vector<vector<MatrixXd> >& batch_feature_maps, MatrixXd& feature_vectors)
{
	int dim = batch_feature_maps[0].size();
	int num_examples = batch_feature_maps.size() * batch_feature_maps[0][0].rows() * batch_feature_maps[0][0].cols();
	feature_vectors.resize(dim, num_examples);

    int k, l;
	for(int i = 0; i < batch_feature_maps.size(); i++)
	{
		for(int j = 0; j < batch_feature_maps[i].size(); j++)
		{
//            #pragma omp parallel for schedule(dynamic,1) collapse(2) private(k,l)
			for(k = 0; k < batch_feature_maps[i][j].rows(); k++)
			{
				for(l = 0; l < batch_feature_maps[i][j].cols(); l++)
				{				
//					cout << "K " << k << endl;
//					cout << "L " << l << endl;	
					int row = j;
					int col = i * batch_feature_maps[i][j].rows() * batch_feature_maps[i][j].cols() + k * batch_feature_maps[i][j].cols() + l;
					feature_vectors(row, col) = batch_feature_maps[i][j](k,l);
				}
			}
		}
	}
	#ifdef GET_FV
	for(int i = 0; i < batch_feature_maps.size(); i++)
	{
		for(int j = 0; j < batch_feature_maps[i].size(); j++)
		{
			cout << "The " << j << "-th feature map of training example " << i << "\n" << batch_feature_maps[i][j] << endl;
		}
	}
	cout << "Feature vectors extracted:\n " << feature_vectors << endl;
	#endif
}		/* -----  end of function ConvPoolLayer::conv2d_activate  ----- */

/* 
 * ===  FUNCTION  ======================================================================
 *         Name: ConvPoolLayer::softmax2conv
 *  Description: This function is part of back propogation, error passed from softmax layer back to convolutional layers
 *    Arguments: err_softmax: gradient at softmax layer
 *                batch_size: batch size
 *                 map_hight: height of an image
 *                 map_width: width of an image
 *                  err_conv: corresponding grdient passed back to convolutional layers
 * =====================================================================================
 */
void softmax2conv(const MatrixXd& err_softmax, const int& batch_size, const int& map_hight, const int& map_width, vector<vector<MatrixXd> >& err_conv)
{
	err_conv.clear();
	int dim = err_softmax.rows();    
	int step = map_hight * map_width;
	int num_examples = err_softmax.cols() / step;
	err_conv.reserve(num_examples);

    int j;

	for(int i = 0; i < num_examples; i++)
	{
		vector<MatrixXd> feature_maps_temp;
        feature_maps_temp.reserve(dim);
		MatrixXd feature_map_temp(1, step);
		for(j = 0; j < dim; j++)
		{
			feature_map_temp = err_softmax.block(j, i * step, 1, step);
			feature_map_temp.resize(map_width, map_hight); 
			feature_maps_temp.push_back(feature_map_temp.transpose());
		}
		err_conv.push_back(feature_maps_temp);
	}

	
	#ifdef SFT_CONV
	cout << "Input error matrix:\n" << err_softmax << endl;
	for(int i = 0; i < err_conv.size(); i++)
	{
		for(int j = 0; j < err_conv.size(); j++)
		{
			cout << "The " << j << "-th feature map of training example " << i << "\n" << err_conv[i][j] << endl;
		}
	}
	#endif
}

void ConvPoolLayer::bp_weight(const vector<MatrixXd>& grad_conved, const vector<MatrixXd>& maps_input, const int& weight_size, vector<MatrixXd>& grad_weight_single)
{
	grad_weight_single.clear();
	MatrixXd grad_weight_temp(weight_size, weight_size);
	MatrixXd grad_weight_partial;

	int num_input = maps_input.size();
	int num_kernel = grad_conved.size() / num_input;

	grad_weight_single.reserve(num_kernel);

	for(int i = 0; i < num_kernel; i++)
	{
		grad_weight_partial.setZero(weight_size, weight_size);
		for(int j = 0; j < num_input; j++)
		{
			int index = i * num_input + j;
			conv2d(maps_input[j], grad_conved[index], grad_weight_temp);
			grad_weight_partial += grad_weight_temp;					
		}
		grad_weight_single.push_back(grad_weight_partial);
	}
}

void ConvPoolLayer::bp_bias(const vector<MatrixXd>& grad_conved, const int& num_kernel, MatrixXd& grad_bias_single)
{
	grad_bias_single.setZero(num_kernel, 1);
	double grad_bias_partial;
	int num_input = grad_conved.size() / num_kernel;
	for(int i = 0; i < num_kernel; i++)
	{
		for(int j = 0; j < num_input; j++)
		{
			int index = i * num_input + j;
			grad_bias_partial = grad_conved[index].sum();
			grad_bias_single(i, 0) += grad_bias_partial;
		}
	}
}

void ConvPoolLayer::bp_input(const vector<MatrixXd>& grad_conved, const vector<MatrixXd>& weight, const int& input_channel, const int& kernel_size, vector<MatrixXd>& grad_input)
{
	grad_input.clear();
	grad_input.reserve(input_channel);

	int hight = grad_conved[0].rows();
	int width = grad_conved[0].cols();

	MatrixXd grad_conved_temp;
	MatrixXd grad_input_temp;
	MatrixXd grad_input_partial;
	grad_input_partial.setZero(hight, width);

	grad_conved_temp.setZero(hight + kernel_size - 1, width + kernel_size - 1);
	int num_kernel = grad_conved.size() / input_channel;

	for(int i = 0; i < input_channel; i++)
	{
		grad_input_partial.setZero(hight, width);
		for(int j = 0; j < num_kernel; j++)
		{
			int index = j * input_channel + i;
			grad_conved_temp.block((kernel_size - 1) / 2, (kernel_size - 1) / 2, hight, width) = grad_conved[index];
			flip_conv2d(grad_conved_temp, weight[j], grad_input_temp);
			grad_input_partial += grad_input_temp;
		}
		grad_input.push_back(grad_input_partial);
	}
}

ConvPoolLayer::ConvPoolLayer(const vector<vector<MatrixXd> >& input, const int& num_kerns, const int& kern_size, const string& conv_mode, const string& act_type)
{
	this->activation_type = act_type;
	if(conv_mode == "valid")
	{	
		this->batch_maps_input = input;
	}

	else if (conv_mode == "same")
	{
		MatrixXd map_input_temp;
		vector<MatrixXd> maps_input_temp;

		for(int i = 0; i < input.size(); i++)
		{
			maps_input_temp.clear();

			for(int j = 0; j < input[i].size(); j++)
			{
				map_input_temp.setZero(input[i][j].rows() + kern_size - 1, input[i][j].cols() + kern_size - 1);
				map_input_temp.block((kern_size - 1) / 2, (kern_size - 1) / 2, input[i][j].rows(), input[i][j].cols()) = input[i][j];
				maps_input_temp.push_back(map_input_temp);
			}
			
			this->batch_maps_input.push_back(maps_input_temp);			
		}				
	}


//	srand((unsigned)time(NULL));

	this->bias = 0.4 * MatrixXd::Random(num_kerns,1);	

	for(int i = 0; i < num_kerns; i++)
	{
//		srand((unsigned)time(NULL));
		MatrixXd weight_temp = 0.4 * MatrixXd::Random(kern_size, kern_size);
		this->weight.push_back(weight_temp);
	}

	compute();

	#ifdef CONVPOOL
	for(int i = 0; i < input.size(); i++)
	{
		for(int j = 0; j < input[i].size(); j++)
		{
			cout << "The " << j << "-th input channel of training example " << i << endl;
			cout << this->batch_maps_input[i][j] << endl;
		}
	}
	cout << "Activation function: " << this->activation_type << endl;
	for(int i = 0; i < this->weight.size(); i++)
	{
		cout << "The " << i << "-th kernel:\n" << this->weight[i] << endl;
	}
	cout << "Bias:\n" << this->bias << endl;
	for(int i = 0; i < this->batch_maps_activated.size(); i++)
	{
		for(int j = 0; j < this->batch_maps_activated[i].size(); j++)
		{
			cout << "The " << j << "-th activated map of training example " << i << endl;
			cout << this->batch_maps_activated[i][j] << endl;
		}
	}
	#endif
}

ConvPoolLayer::ConvPoolLayer(const vector<vector<MatrixXd> >& input, const int& num_kerns, const int& kern_size, const string& conv_mode, const string& act_type, const vector<MatrixXd>& weight_, const MatrixXd& bias_)
{
	this->activation_type = act_type;
	if(conv_mode == "valid")
	{	
		this->batch_maps_input = input;
	}

	else if (conv_mode == "same")
	{
		MatrixXd map_input_temp;
		vector<MatrixXd> maps_input_temp;

		for(int i = 0; i < input.size(); i++)
		{
			maps_input_temp.clear();

			for(int j = 0; j < input[i].size(); j++)
			{
				map_input_temp.setZero(input[i][j].rows() + kern_size - 1, input[i][j].cols() + kern_size - 1);
				map_input_temp.block((kern_size - 1) / 2, (kern_size - 1) / 2, input[i][j].rows(), input[i][j].cols()) = input[i][j];
				maps_input_temp.push_back(map_input_temp);
			}
			
			this->batch_maps_input.push_back(maps_input_temp);			
		}				
	}

	this->bias = bias_;
	this->weight = weight_;

	compute();

}

void ConvPoolLayer::compute()
{
	this->batch_maps_activated.clear();
	this->batch_maps_activated.reserve(this->batch_maps_input.size());

	vector<MatrixXd> maps_activated_temp;
	vector<MatrixXd> maps_output_temp;
	vector<vector<MatrixXd> > coordinates_temp;

	for(int i = 0; i < this->batch_maps_input.size(); i++)
	{
		maps_activated_temp.clear();
		maps_output_temp.clear();
		coordinates_temp.clear();

        struct timeval tim;
        gettimeofday(&tim, NULL);
        double t1=tim.tv_sec+(tim.tv_usec/1000000.0);
        // TODO: GPU Version
        conv2d_activate(this->batch_maps_input[i], this->weight, this->bias, this->activation_type, maps_activated_temp);
        //
        gettimeofday(&tim, NULL);
        double t2=tim.tv_sec+(tim.tv_usec/1000000.0);
//        printf("%.6lf seconds elapsed conv2d_activate()\n", t2-t1);

		this->batch_maps_activated.push_back(maps_activated_temp);
        gettimeofday(&tim, NULL);
        t1=tim.tv_sec+(tim.tv_usec/1000000.0);
	}
	
}

void ConvPoolLayer::bp_activate(const vector<vector<MatrixXd> >& grad_batch_maps_activated, const vector<vector<MatrixXd> >& batch_maps_activated, vector<vector<MatrixXd> >& grad_batch_maps_conved)
{
	grad_batch_maps_conved.clear();
	grad_batch_maps_conved.reserve(grad_batch_maps_activated.size());
	int row = grad_batch_maps_activated[0][0].rows();
	int col = grad_batch_maps_activated[0][0].cols();
	MatrixXd grad_map_conved(row, col);
	vector<MatrixXd> vec_grad;
	for(int i = 0; i < grad_batch_maps_activated.size(); i++)
	{
		vec_grad.clear();
        vec_grad.reserve(grad_batch_maps_activated[i].size());
		for(int j = 0; j < grad_batch_maps_activated[i].size(); j++)
		{

			#ifdef BP_ACT
			cout << "The " << j << "-th activated map of training example " << i << endl;
			cout << batch_maps_activated[i][j] << endl;
			cout << "The " << j << "-th activated gradient map of training example " << i << endl;
			cout << grad_batch_maps_activated[i][j] << endl;
			#endif

			for(int k = 0; k < grad_batch_maps_activated[i][j].rows(); k++)
			{
				for(int l = 0; l < grad_batch_maps_activated[i][j].cols(); l++)
				{
					grad_map_conved(k, l) = grad_batch_maps_activated[i][j](k, l) * (1 - pow(batch_maps_activated[i][j](k, l), 2));
				}
			}	
			
			#ifdef BP_ACT
			cout << "The " << j << "-th conved gradient map of training example " << i << endl;
			cout << grad_map_conved << endl;
			#endif
			
			vec_grad.push_back(grad_map_conved);
		}
		grad_batch_maps_conved.push_back(vec_grad);
	}	
}

void ConvPoolLayer::bp_conv(const vector<vector<MatrixXd> >& grad_batch_maps_conved, const double& lr)
{
//	 compute grads with respect to weights
	this->grad_weight.clear();
	this->grad_weight.reserve(this->weight.size());

	vector<MatrixXd> grad_weight_single;
	MatrixXd grad_init = MatrixXd::Zero(this->weight[0].rows(), this->weight[0].cols());
	int weight_size = this->weight[0].rows();
	for(int i = 0; i < this->weight.size(); i++)
	{
		this->grad_weight.push_back(grad_init);
	}		

	for(int i = 0; i < grad_batch_maps_conved.size(); i++)
	{
		bp_weight(grad_batch_maps_conved[i], this->batch_maps_input[i], weight_size, grad_weight_single);
		for(int j = 0; j < this->weight.size(); j++)
		{
			this->grad_weight[j] += grad_weight_single[j];
		}
	}

//	 compute grads with respect to bias
	MatrixXd grad_bias_single;
	this->grad_bias.setZero(this->bias.rows(), this->bias.cols());
	for(int i = 0; i < grad_batch_maps_conved.size(); i++)
	{
		bp_bias(grad_batch_maps_conved[i], this->weight.size(), grad_bias_single);
		this->grad_bias += grad_bias_single;
	}

//	 compute grads with respect to inputs
	this->grad_batch_maps_input.clear();
	this->grad_batch_maps_input.reserve(this->batch_maps_input.size());
	vector<MatrixXd> grad_input;
	int num_channel = this->batch_maps_input[0].size();
	for(int i = 0; i < grad_batch_maps_conved.size(); i++)
	{
		bp_input(grad_batch_maps_conved[i], this->weight, num_channel, weight_size, grad_input);
		this->grad_batch_maps_input.push_back(grad_input);
	}
	
//	compute new weight and bias
	for(int i = 0; i < this->grad_weight.size(); i++)
	{
		this->weight[i] = this->weight[i] - lr * this->grad_weight[i];         // new weight
	}	
	this->bias = this->bias - lr * this->grad_bias;                      // new bias
}

void ConvPoolLayer::back_prop(const vector<vector<MatrixXd> >& grad_batch_maps_activated, const double& lr)
{
	vector<vector<MatrixXd> > gradient_convolved;
	bp_activate(grad_batch_maps_activated, this->batch_maps_activated, gradient_convolved);
	bp_conv(gradient_convolved, lr);
}

}
