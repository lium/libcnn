/*
 * =====================================================================================
 *
 *       Filename:  yaml_t_load.cpp
 *
 *    Description:  yaml_test test program
 *
 *        Version:  1.0
 *        Created:  Thursday, October 23, 2014 11:49:07 HKT
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <iostream>
#include <fstream>
#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <stdlib.h>
#include <string>
#include <string.h>

using namespace std;
using namespace Eigen;
using namespace YAML;
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  
 * =====================================================================================
 */
string dir = "../seq.yaml";
MatrixXd d(3,3);
int Label_number;
VectorXi v(3);
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
	
	

		
int main ( int argc, char *argv[] )
{
	//Node node = LoadFile("seq.yaml");
	//MatrixXd d(3,3);
	//for(unsigned i = 0; i < 3; i++){
	//	for(unsigned j = 0; j < 3; j++){
	//		d(i,j) = node["cnn_output"][3 * i + j].as<double>();
	//	} 
	//}
	//cout << d << endl;		
	//int Label_number = node["Label_number"].as<int>();
	//cout << Label_number << endl;
	//VectorXi v(3);
	//for(unsigned i = 0; i < 3; i++){
	//	v(i) = node["ground_truth"][i].as<int>();
	//}
	//cout << v << endl;	
	//double w_bound = node["w_bound"].as<double>();
	//double b_bound = node["b_bound"].as<double>();
	//double learning_rate = node["learning_rate"].as<double>();
	//cout << w_bound << endl;
	//cout << b_bound << endl;
	//cout << learning_rate << endl;
	parser_yaml_para(dir,d,Label_number,v,w_bound,b_bound,learning_rate);
	cout << d << endl;
	return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
