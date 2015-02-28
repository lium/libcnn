#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <math.h>
#include "softmax_class.h"
using namespace std;
using namespace Eigen;
namespace libcnn{

    Softmax::Softmax(const MatrixXd & cnn_output, const int & Label_number, const VectorXi & ground_truth, const double & w_bound, const double & b_bound){
        srand((unsigned)time(NULL));
        weight = w_bound * MatrixXd::Random(Label_number,cnn_output.rows()); 
        bias = b_bound * MatrixXd::Random(Label_number,1);
        input = cnn_output;
        truth = ground_truth;
        m = VectorXd::Zero(input.cols());
        c = VectorXd::Zero(input.cols());
        prob = MatrixXd::Zero(input.rows(), input.cols());
    }

    void Softmax::calculation_output(){
        MatrixXd A(weight.rows(), input.cols());
        size_t j,i,k;
#pragma omp parallel for schedule(dynamic,1) collapse(2) private(i,j)
        for (i = 0 ; i < weight.rows(); i++){
            for (j = 0; j < input.cols(); j++){
                A(i,j) = exp((weight.row(i)).dot(input.col(j)) + bias(i,0));
            }
        }

        VectorXf::Index maxIndex;
#pragma omp parallel for private(k)
        for(k = 0 ; k < A.cols(); k++){
            assert(A.col(k).sum() != 0);
            A.col(k) = A.col(k).array() / A.col(k).sum();
            double max = A.col(k).maxCoeff(&maxIndex);
            m(k) = maxIndex ;
            c(k) = log(A(truth(k), k));
        }     

        prob = A;
        cost = (-1) * c.sum() / input.cols();
    }

    Softmax::Softmax(const MatrixXd & cnn_output, const int & Label_number, const VectorXi & ground_truth, const double & w_bound, const double & b_bound, const MatrixXd & initial_weight, const MatrixXd & initial_bias){
		weight = initial_weight;
		bias = initial_bias;
        input = cnn_output;
        truth = ground_truth;
        m = VectorXd::Zero(input.cols());
        c = VectorXd::Zero(input.cols());
        prob = MatrixXd::Zero(input.rows(), input.cols());
    }

    void Softmax::backprop(const double& lr){
        double coe = double(-1.0 / input.cols()) ;
        MatrixXd grad_w(weight.rows(), weight.cols()), grad_b(bias.rows(), bias.cols());
        grad_input.resize(input.rows(), input.cols());
        MatrixXd truth_matrix(weight.rows(), input.cols()), truth_m_prob(weight.rows(), input.cols());
        truth_matrix = MatrixXd::Zero(weight.rows(),input.cols());
        truth_m_prob = MatrixXd::Zero(weight.rows(),input.cols());
        grad_w = MatrixXd::Zero(weight.rows(),weight.cols());
        grad_b = MatrixXd::Zero(bias.rows(), bias.cols());
        grad_input = MatrixXd::Zero(input.rows(), input.cols());

        size_t i, j, k;
        for(i = 0; i < input.cols(); i++){
            truth_matrix(truth(i), i) = 1;
        }     
        truth_m_prob = truth_matrix - prob;

        for(j = 0; j < truth_m_prob.rows(); j++){
//#pragma omp parallel for private(k)
            for(k = 0; k < input.cols(); k++){
                grad_w.row(j) += (input.col(k).array() * truth_m_prob(j,k)).matrix(); 
            }
        }

        for(i = 0; i < truth_m_prob.cols(); i++){
//#pragma omp parallel for private(j)
            for(j = 0; j < weight.rows(); j++){
                grad_input.col(i) += (weight.row(j).array() * truth_m_prob(j,i)).matrix();
            }
        }

        grad_w = coe * grad_w;
        grad_input = coe * grad_input;
        grad_b = coe * truth_m_prob.rowwise().sum(); 

        weight = weight - lr * grad_w;
        bias  = bias - lr * grad_b;
    } 

}

