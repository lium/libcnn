#include <iostream>
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include "conv_layer.h"

using namespace std;
using namespace libcnn;
using namespace Eigen;

TEST(SigmoidTest, AssertionNear)
{
	ASSERT_NEAR(0.5, sigmoid(0), 0.001);
	ASSERT_NEAR(0.9999546, sigmoid(10), 0.001);
	ASSERT_NEAR(0.0474258, sigmoid(-3), 0.0001);
}

TEST(GetFeatureVectorTest, AssertionEqual)
{
	MatrixXd fm1(2,2), fm2(2,2), fm3(2,2), fm4(2,2);
	fm1 << 1,2,3,4; fm2 << 5,6,7,8; fm3 << 9,10,11,12; fm4 << 13,14,15,16;
	vector<MatrixXd> vec_fm1, vec_fm2;
	vec_fm1.push_back(fm1); vec_fm1.push_back(fm2);
	vec_fm2.push_back(fm3); vec_fm2.push_back(fm4);
	vector<vector<MatrixXd> > vec_vec_fm1;
	vec_vec_fm1.push_back(vec_fm1); vec_vec_fm1.push_back(vec_fm2);

	MatrixXd fm5(3,3), fm6(3,3), fm7(3,3), fm8(3,3), fm9(3,3), fm10(3,3);
	fm5 << 1,2,3,4,5,6,7,8,9; fm6 << 2,3,4,5,6,7,8,9,0; fm7 << 6,7,8,9,0,1,2,3,4;
	fm8 << 9,8,7,6,5,4,3,2,1; fm9 << 5,6,7,8,9,0,1,2,3; fm10 << 7,6,5,4,3,2,1,0,9;
	vector<MatrixXd> vec_fm3, vec_fm4;
	vec_fm3.push_back(fm5); vec_fm3.push_back(fm6); vec_fm3.push_back(fm7);
	vec_fm4.push_back(fm8); vec_fm4.push_back(fm9); vec_fm4.push_back(fm10);
	vector<vector<MatrixXd> > vec_vec_fm2;
	vec_vec_fm2.push_back(vec_fm3); vec_vec_fm2.push_back(vec_fm4);

	MatrixXd out1, out2;
	MatrixXd exp1(2,8), exp2(3,18);
	exp1 << 1,2,3,4,9,10,11,12,5,6,7,8,13,14,15,16;
	exp2 << 1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,0,5,6,7,8,9,0,1,2,3,6,7,8,9,0,1,2,3,4,7,6,5,4,3,2,1,0,9;
	
	get_feature_vector(vec_vec_fm1, out1);
	get_feature_vector(vec_vec_fm2, out2);

	ASSERT_EQ(exp1, out1);
	ASSERT_EQ(exp2, out2);


}

int main(int argc, char **argv)
{
	
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();

	return 0;
}
