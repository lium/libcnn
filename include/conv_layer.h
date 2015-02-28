#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace Eigen;
namespace libcnn{

inline double sigmoid(const double& x)
{
	return 1.0 / (1 + exp(-x));
}

void get_feature_vector(const vector<vector<MatrixXd> >&, MatrixXd&);
void softmax2conv(const MatrixXd&, const int&, const int&, const int&, vector<vector<MatrixXd> >&);


class ConvPoolLayer
{
	public:
		vector<vector<MatrixXd> > batch_maps_input;
		vector<vector<MatrixXd> > grad_batch_maps_input;
		vector<MatrixXd> weight;
		vector<MatrixXd> grad_weight;
		MatrixXd bias;
		MatrixXd grad_bias;
		vector<vector<MatrixXd> > batch_maps_activated;
		vector<vector<vector<MatrixXd> > > batch_coordinates;
		string activation_type;

		ConvPoolLayer(const vector<vector<MatrixXd> >&, const int&, const int&, const string&, const string&); 
		ConvPoolLayer(const vector<vector<MatrixXd> >&, const int&, const int&, const string&, const string&, const vector<MatrixXd>&, const MatrixXd&);
		void compute();
		
		void bp_activate(const vector<vector<MatrixXd> >&, const vector<vector<MatrixXd> >&, vector<vector<MatrixXd> >&);
		void bp_conv(const vector<vector<MatrixXd> >&, const double&);
		void back_prop(const vector<vector<MatrixXd> >&, const double&);

	private:
		void conv2d(const MatrixXd&, const MatrixXd&, MatrixXd&);
		void conv2d(const vector<MatrixXd>&, const vector<MatrixXd>, vector<MatrixXd>&); 
		void flip_conv2d(const MatrixXd&, const MatrixXd&, MatrixXd&);

		void activate(const MatrixXd&, const double&, MatrixXd&);
		void activate(const vector<MatrixXd>&, const MatrixXd&, vector<MatrixXd>&);

		void conv2d_activate(const MatrixXd&, const MatrixXd&, const double&, const string&, MatrixXd&);
		void conv2d_activate(const vector<MatrixXd>&, const vector<MatrixXd>&, const MatrixXd&, const string&, vector<MatrixXd>&);

		void bp_weight(const vector<MatrixXd>&, const vector<MatrixXd>&, const int&, vector<MatrixXd>&);
		void bp_input(const vector<MatrixXd>&, const vector<MatrixXd>&, const int&, const int&, vector<MatrixXd>&);
		void bp_bias(const vector<MatrixXd>&, const int&, MatrixXd&);

};

}

#endif

