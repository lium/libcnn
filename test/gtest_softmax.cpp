/*
 * =====================================================================================
 *
 *       Filename:  gtest_softmax.cpp
 *
 *    Description:  gtest_softmax.cpp
 *
 *        Version:  1.0
 *        Created:  Sunday, October 19, 2014 05:31:48 HKT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <Eigen/Dense>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <gtest/gtest.h>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include <string>
#include <string.h>
#include "../include/softmax_class.h"
using namespace std;
using namespace Eigen;
using namespace libcnn;
using namespace YAML;

string dir = string ("../seq.yaml");
MatrixXd cnn_output(3,3);
int Label_number;
VectorXi ground_truth(3);
double w_bound;
double b_bound;
double learning_rate;

void parser_yaml_para(string & dir, MatrixXd & d, int & Label_number, VectorXi & v, double & w_bound, double & b_bound, double & learning_rate){
	Node node = LoadFile(dir);
	for(unsigned i = 0; i < 3; i++){
		for(unsigned j = 0; j < 3; j++){
			d(i,j) = node["cnn_output"][3 * i + j].as<double>();
		} 
	}
	Label_number = node["Label_number"].as<int>();
	
	for(unsigned i = 0; i < 3; i++){
		v(i) = node["ground_truth"][i].as<int>();
	}
	
	w_bound = node["w_bound"].as<double>();
	b_bound = node["b_bound"].as<double>();
	learning_rate = node["learning_rate"].as<double>();
}

TEST(IntTest, IntNumber){
	parser_yaml_para(dir,cnn_output,Label_number,ground_truth,w_bound,b_bound,learning_rate);
	//Softmax classifier(cnn_output, Label_number, ground_truth, w_bound, b_bound, learning_rate);
	//classifier.calculation_output();
	double lr_to_be_test = 0.1;
	ASSERT_EQ(lr_to_be_test, learning_rate);
}
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  
 * =====================================================================================
 */
int main ( int argc, char *argv[] )
{
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}				/* ----------  end of function main  ---------- */

