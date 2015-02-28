// Name        : test_opencv.cpp

#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <vector>
#include <string>
#include <map>
// NOTE: include Eigen before opencv2/core/eigen.hpp or
// code will fail to compile
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::VectorXf;
using Eigen::VectorXd;

typedef Eigen::Matrix<uchar, Dynamic, Dynamic> MatrixXc;
typedef map<string,Mat> SrcImgMapType;
typedef map<string,VectorXf> SrcDepthMapType;
typedef map<string,VectorXi> SrcLabelMapType;

/*
 * Helper function to read a *.txt of filenames
 * and pushes filenames to a vector<string>
 */
void getNameListInFile (string pathToFile, vector<string> &vNameList) {
  if (!vNameList.empty())
      vNameList.clear();
  ifstream ifs (pathToFile.c_str());
  if (!ifs.is_open()) {
    cout << "Cannot open file at " << pathToFile << endl;
    return;
  }
  string line;
  while (getline(ifs, line)) {
    vNameList.push_back(line);
  }
  ifs.close();
}

/* Helper function taking a list of valid full filenames
 * (nameList) to form vector< vector <MatrixXi> >
 */
void buildImageSubset(string pathToImageDir,
                      const vector<string> &nameList,
                      vector< vector<MatrixXi> > &subsetImg) {
  if (!subsetImg.empty())
      subsetImg.clear();
  // read all images
  for (unsigned int fn = 0; fn < nameList.size(); fn++) {
    string fullpath = pathToImageDir + nameList[fn] + ".jpg";
    // read the image
    Mat bgr = imread(fullpath), yuv;
    if (bgr.empty()) {
      cout << nameList[fn] + ".jpg" << " not found in " << pathToImageDir << endl;
      continue; // jump to next loop
    }
    cvtColor(bgr, yuv, CV_BGR2YCrCb);
    // split into individual channels
    vector<Mat> vMatSplit;
    split(yuv, vMatSplit);
    vector<MatrixXi> vEigSrcImgSplit(vMatSplit.size());
    // copy channels from split Mat to vector of MatrixXi
    for (unsigned int ch = 0; ch < vMatSplit.size(); ch++) {
      cv2eigen<int>(vMatSplit[ch], vEigSrcImgSplit[ch]);
    }
    subsetImg.push_back(vEigSrcImgSplit);
  }
}

/* Helper function taking a list of filenames
 * (nameList) and a SrcImgMapType (mapImg) to form
 * vector< vector <MatrixXi> >
 */
void buildRegionSubset(string pathToLabelDir,
                      const vector<string> &nameList,
//                    vector<VectorXi> &subsetLabel) {
                      VectorXi &subsetLabel) {
  vector<int> data;
  // read all label files
  for (unsigned int fn = 0; fn < nameList.size(); fn++) {
    string fullpath = pathToLabelDir + nameList[fn] + ".regions.txt";
    ifstream ifs (fullpath.c_str());
    if (!ifs.is_open()) {
      cout << nameList[fn]+".regions.txt" << " not found in " << pathToLabelDir << endl;
      continue; // jump to next loop
    }
    string line;
    while (getline(ifs, line)) {
      unsigned long idx = 0;
      while ( (idx = line.find(" ")) != string::npos) {
        data.push_back(atoi(line.substr(0,idx).c_str()));
        line.erase(0, idx+1);
      } // last data remaining
      data.push_back(atoi(line.substr(0,idx).c_str()));
    }
    // now all integers from a .depth.txt is in a vector
    /*
    Eigen::Map<VectorXi> v (data.data(), data.size(), 1);
    VectorXi tmp = v;
    subsetLabel.push_back(tmp);
    */
  }
  //  Eigen::Map<VectorXi> v (data.data(), data.size(), 1);
  subsetLabel = Eigen::Map<VectorXi> (data.data(), data.size(), 1);
;
}

void mergeVectors (const vector<VectorXi> &src,
                        VectorXi &dest) {
  unsigned long rows = 0;
  for (unsigned int i = 0; i < src.size(); i++) {
    rows += src[i].rows();
  }
  cout << "Debug: rows == " << rows << " size == " << src.size() 
       << endl;
  dest.resize(rows,1);
  for (unsigned int i = 0; i < src.size(); i++) {
    dest << src[i];
  }
}

void buildImageSet (
  string pathToImageDir,
  const vector<string> &vNameListEval,
  const vector<string> &vNameListTrain,
  vector < vector<MatrixXi> > &imgSetEval,
  vector < vector<MatrixXi> > &imgSetTrain ) {

  if(!imgSetEval.empty()) {
    imgSetEval.clear();
  }
  if(!imgSetTrain.empty()) {
    imgSetTrain.clear();
  }
  /*
   * Read and convert cv::Mat to two vectors of vectors of
   * Eigen::Matrix<int, Dynamic, Dynamic> (i.e. MatrixXi)
   */
  buildImageSubset(pathToImageDir, vNameListEval, imgSetEval);
  // build vvEigSrcImgTrain
  buildImageSubset(pathToImageDir, vNameListTrain, imgSetTrain);
  cout << "Eval has " << imgSetEval.size() << " images." << endl;
  cout << "Train has " << imgSetTrain.size() << " images." << endl;
}

void buildRegionSet (
  string pathToLabelDir,
  const vector<string> &vNameListEval,
  const vector<string> &vNameListTrain,
  VectorXi &labelSetEval,
  VectorXi &labelSetTrain) {

  vector<VectorXi> vLabelSetEval, vLabelSetTrain;
  //  buildRegionSubset(pathToLabelDir, vNameListEval, vLabelSetEval);
  //  buildRegionSubset(pathToLabelDir, vNameListTrain, vLabelSetTrain);
  //  cout << "Built vector of VectorXi, merging..." << endl;
  //  mergeVectors(vLabelSetEval, labelSetEval);
  //  mergeVectors(vLabelSetTrain, labelSetTrain);
  //  cout << "Eval has " << vLabelSetEval.size() << " region files." << endl;
  //  cout << "Train has " << vLabelSetTrain.size() << " region files." << endl;
  buildRegionSubset(pathToLabelDir, vNameListEval, labelSetEval);
  buildRegionSubset(pathToLabelDir, vNameListTrain, labelSetTrain);
}

void buildDataSet (string basePath,
  vector< vector<MatrixXi> > &imgSetEval,
  vector<string> &vFilenameEval,
  VectorXi &labelSetEval,
  vector< vector<MatrixXi> > &imgSetTrain,
  vector<string> &vFilenameTrain,
  VectorXi &labelSetTrain ) {

  const string IMAGE_SUBDIR = "images/";
  const string LABEL_SUBDIR = "labels/";
  const string EVAL_FILENAME = "evalList.txt";
  //const string EVAL_FILENAME = "evalListTmp.txt";
  const string TRAIN_FILENAME = "trainList.txt";
  //const string TRAIN_FILENAME = "trainListTmp.txt";
  string pathToImageDir = basePath + IMAGE_SUBDIR;
  string pathToLabelDir = basePath + LABEL_SUBDIR;
  string filenameEval   = basePath + EVAL_FILENAME;
  string filenameTrain  = basePath + TRAIN_FILENAME;

  // Get list of filenames in evalList.txt and trainList.txt
  vector<string> vNameListEval, vNameListTrain;
  getNameListInFile(filenameEval, vNameListEval);
  getNameListInFile(filenameTrain, vNameListTrain);

  //Debug
  /*
  for (int i = 0; i < vNameListEval.size(); i++) {
    cout << "Eval " << i << ": " << vNameListEval[i] << endl;
  }
  for (int i = 0; i < vNameListTrain.size(); i++) {
    cout << "Train " << i << ": " << vNameListTrain[i] << endl;
  }*/

  // Build vector < vector <MatrixXi> > of images, both eval and train
  buildImageSet (pathToImageDir,
    vNameListEval,  vNameListTrain,
    imgSetEval,     imgSetTrain );
  cout << "Size of Eval and Train image sets: " << imgSetEval.size()
       << " and " << imgSetTrain.size() << endl;
  // Build VectorXi of region labels, both eval and train
  buildRegionSet (pathToLabelDir,
    vNameListEval,  vNameListTrain,
    labelSetEval,   labelSetTrain);
  cout << "Size of Eval and Train label sets: " << labelSetEval.rows()
       << " and " << labelSetTrain.rows() << endl;
}

void convertEigYuvToCvBgr (const vector<MatrixXi> &vEigYuv, Mat &matBgr) {
  vector<Mat> vMatYuv (vEigYuv.size());
  for (unsigned int ch = 0; ch < vEigYuv.size(); ch++) {
    MatrixXc tmp = vEigYuv[ch].cast<uchar>();
    eigen2cv<uchar>(tmp, vMatYuv[ch]);
  }
  Mat matYuv;
  merge(vMatYuv, matYuv);
  cvtColor(matYuv, matBgr, CV_YCrCb2BGR);
}

void convertLabelToImage (const VectorXi &label, long pos, long length, Mat &img) {
  img = Mat(320, 240, CV_8UC3);
  for (long i = pos; i < pos+length; i++) {
    switch ((int)label.rows()) {
    case -1:
      img.at<Vec3b>(i) = Vec3b(0,0,0);  // black
      break;
    case 0:
      img.at<Vec3b>(i) = Vec3b(255,0,0); // blue
      break;
    case 1:
      img.at<Vec3b>(i) = Vec3b(0,255,0); // green
      break;
    case 2:
      img.at<Vec3b>(i) = Vec3b(0,0,255); // red
      break;
    case 3:
      img.at<Vec3b>(i) = Vec3b(255,255,0); // cyan
      break;
    case 4:
      img.at<Vec3b>(i) = Vec3b(255,0,255); // purple
      break;
    case 5:
      img.at<Vec3b>(i) = Vec3b(0,255,255); // yellow
      break;
    case 6:
      img.at<Vec3b>(i) = Vec3b(150,150,150); //gray
      break;
    case 7:
      img.at<Vec3b>(i) = Vec3b(255,255,255); // white
      break;
    } // end switch
  }
}

void help() {
  cout << "Expected 1 argument: " 
       << "path to root directory containing " << endl
       << "  images/" << endl
       << "  labels/" << endl
       << "  evalList.txt" << endl
       << "  trainList.txt" << endl;
}

/*
 * As a usage reference for buildDataSet()
 */
int main(int argc, char **argv) {
  /*
  vector< vector<MatrixXi> > vvEigSrcImgEval, vvEigSrcImgTrain;
  //cout << "Checking: " << vvEigSrcImgEval.size() << " " << vvEigSrcImgTrain.size() << endl;
  Mat test1, test2;
  convertEigYuvToCvBgr(vvEigSrcImgEval[0], test1);
  convertEigYuvToCvBgr(vvEigSrcImgTrain[0], test2);
  imshow("Test Eval", test1);
  imshow("Test Train", test2);
  waitKey();
  */
  if (argc != 2) {
    help();
    return -1;
  }

  vector< vector<MatrixXi> > imgSetEval, imgSetTrain;
  vector<string> vFilenameEval, vFilenameTrain;
  VectorXi labelSetEval, labelSetTrain;
  buildDataSet (argv[1],
                imgSetEval,
                vFilenameEval,
                labelSetEval,
                imgSetTrain,
                vFilenameTrain,
                labelSetTrain);
  // Debug
  // Here, the output are displays the first images of evalList and trainList
  // and their respective labels into eval.txt and train.txt for checking
  Mat test1, test2, test3, test4;
  convertEigYuvToCvBgr(imgSetEval[0], test1);
  convertEigYuvToCvBgr(imgSetTrain[0], test2);
  ofstream ofs1 ("eval.txt"), ofs2("train.txt");
  for (int i = 0; i < 320; i++) {
    for (int j = 0; j < 239; j++) {
      ofs1 << labelSetEval(i*240+j) << ' ';
      ofs2 << labelSetTrain(i*240+j) << ' ';
    }
    ofs1 << labelSetEval((i+1)*240-1) << endl;
    ofs2 << labelSetTrain((i+1)*240-1) << endl;
  }
  ofs1.close(); ofs2.close();
  cout << "Note: eval.txt and train.txt written" << endl;
  imshow("test1", test1);
  imshow("test2", test2);
  moveWindow("test2", test1.cols+5, 0); 
  cout << "Press any key to exit" << endl;
  waitKey();
  return 0;
}

