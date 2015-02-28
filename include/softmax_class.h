	#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <math.h>
using namespace std;
using namespace Eigen;
namespace libcnn{


class Softmax
{
          public:
          Eigen::MatrixXd weight, bias, input, prob;//prob---each column represent softmax probability of training set       
          Eigen::VectorXi truth;// each value represent the truth value of each pixel
          Eigen::VectorXd m, c;//m store the argmax value of prob, c store the log value of prob
          double cost, lr; //lr means the learning rate 
		  Eigen::MatrixXd grad_input;	

          Softmax (const MatrixXd &, const int &, const VectorXi &, const double &, const double &);                             /* constructor */
          Softmax (const MatrixXd &, const int &, const VectorXi &, const double &, const double &, const MatrixXd &, const MatrixXd &);                             /* constructor */
          void calculation_output();
          void backprop(const double &);

};


}
