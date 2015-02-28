#ifndef SAVE_LOAD_H 
#define SAVE_LOAD_H

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
using namespace std;
using namespace Eigen;
namespace libcnn{
void save(const vector<vector<MatrixXd> >&, const vector<MatrixXd>&, const MatrixXd&, const MatrixXd&);
void load(const vector<int>& num_kerns, vector<vector<MatrixXd> >&, vector<MatrixXd>&, MatrixXd&, MatrixXd&);
}

#endif
