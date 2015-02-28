=====================
Expected directory format:
=====================

(At root directory)
images/
labels/
evalList.txt
trainList.txt

=====================
HOW TO RUN:
=====================

cmake ./
make
./build_dataset

=====================
Output Explanation:
=====================

Program reads names of files from evalList.txt and trainList.txt, and 

1) read the corresponding images from images/ and region labels from labels/
2) stores images as vector < vector<MatrixXi> > imageSetEval, imageSetTrain
3) stores ALL labels in a VectorXi labelSetEval, labelSetTrain
4) (Debug) displays images of the first filename in evalList.txt and trainList.txt
5) (Debug) outputs eval.txt and train.txt that are identical to region label files of the images 
