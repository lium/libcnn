#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>

using namespace std;
using namespace Eigen;
namespace libcnn{

void normalize(MatrixXd& );
void data_generation(const int&, const int&, const int&,const int&, vector<vector<MatrixXd> >&, VectorXi&);
void unit_scaling(vector<vector<MatrixXd> >&);
void label_sampling(const int&, const int&, const int&,const int&, VectorXi&, VectorXi&);

}
#endif
