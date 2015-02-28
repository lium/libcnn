#include <iostream>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include "../include/conv_layer.h"

using namespace std;
using namespace libcnn;
using namespace Eigen;

TEST(SigmoidTest, AssertionNear)
{
	ASSERT_NEAR(0.5, sigmoid(0), 0.001);
	ASSERT_NEAR(0.9999546, sigmoid(10), 0.001);
	ASSERT_NEAR(0.0474258, sigmoid(-3), 0.0001);
}

TEST(Conv2dTest, AssertionNear)
{
	MatrixXd map1(4,4), map2(5,5);
	map1 << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
	map2 << 14,15,16,17,18,19,20,21,22,23,24,0,1,2,3,4,5,6,7,8,9,10,11,12,13;
	MatrixXd ker1(3,3), ker2(3,3);
	ker1 << 0,0,1,1,0,0,0,1,0;
	ker2 << 3,1,2,0,2,1,1,1,1;
	MatrixXd out1(2,2), out2(2,2), out3(3,3), out4(3,3);
	out1 << 18, 21, 30, 33;
	out2 << 60, 72, 108, 120;
	out3 << 35, 38, 41, 50, 28, 31, 15, 18, 21;
	out4 << 175, 162, 174, 135, 147, 159, 120, 57, 69;
	MatrixXd est_out1, est_out2, est_out3, est_out4;
	conv2d(map1, ker1, est_out1);
	conv2d(map1, ker2, est_out2);
	conv2d(map2, ker1, est_out3);
	conv2d(map2, ker2, est_out4);

	ASSERT_EQ(out1, est_out1);
	ASSERT_EQ(out2, est_out2);
	ASSERT_EQ(out3, est_out3);
	ASSERT_EQ(out4, est_out4);

	MatrixXd out5(2,2), out6(3,3);
	out5 << 18, 21, 30, 33;
	out6 << 155, 117, 129, 115, 102, 114, 100, 87, 99;
	flip_conv2d(map1, ker1, est_out1);
	flip_conv2d(map2, ker2, est_out2);

	ASSERT_EQ(out5, est_out1);
	ASSERT_EQ(out6, est_out2);

	vector<MatrixXd> maps;
	vector<MatrixXd> kers;
	maps.push_back(map1);
	maps.push_back(map2);
	kers.push_back(ker1);
	kers.push_back(ker2);
	vector<MatrixXd> output;
	conv2d(maps, kers, output);
	
	ASSERT_EQ(out1, output[0]);
	ASSERT_EQ(out2, output[1]);
	ASSERT_EQ(out3, output[2]);
	ASSERT_EQ(out4, output[3]);
}

int main(int argc, char **argv)
{
	
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();

	return 0;
}
