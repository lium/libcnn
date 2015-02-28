#ifndef POOL_H
#define POOL_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <string>

using namespace std;
using namespace Eigen;

namespace libcnn{

class Pool
{
	public:
		vector<vector<MatrixXd> > output_batch_pooled;
		vector<vector<MatrixXd> > grad_batch_input;

		Pool(const vector<vector<MatrixXd> >&, const int&, const int&);
		void compute(const vector<vector<MatrixXd> >&);
		void back_prop(vector<vector<MatrixXd> >&);		

		vector<vector<vector<MatrixXd> > > _coordinates;
	private:
		int _scale_hight, _scale_width;
		vector<vector<double> > vec_vec_std;

		void _pool(const MatrixXd&, const int&, const int&, vector<MatrixXd>&, MatrixXd&);
		void _pool_(const vector<MatrixXd>&, const int&, const int&, vector<vector<MatrixXd> >&, vector<MatrixXd>&);
		void _normalize(MatrixXd&, double&);
		void _normalize_(vector<vector<MatrixXd> >&, vector<vector<double> >&);
		void _denormalize(MatrixXd&, const double&);
		void _denormalize_(vector<vector<MatrixXd> >&, const vector<vector<double> >&);
				
};

}
	
#endif
