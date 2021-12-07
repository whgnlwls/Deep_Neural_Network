#pragma once

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
using namespace std;

#define NUM_OF_DATA 75
#define NUM_OF_IN_NODE 4
#define NUM_OF_HID1_NODE 8
#define NUM_OF_OUT_NODE 3

#define ALPHA 0.01

#define COUNT 100

class AI {
private:
	double trainingDat[NUM_OF_DATA][NUM_OF_IN_NODE];
	double testingDat[NUM_OF_DATA][NUM_OF_IN_NODE];
	double cls1[NUM_OF_OUT_NODE] = { 1, 0, 0 };
	double cls2[NUM_OF_OUT_NODE] = { 0, 1, 0 };
	double cls3[NUM_OF_OUT_NODE] = { 0, 0, 1 };

	double inputDat[NUM_OF_IN_NODE];

	double weight_ItoH1[NUM_OF_IN_NODE][NUM_OF_HID1_NODE];
	double weight_H1toO[NUM_OF_HID1_NODE][NUM_OF_OUT_NODE];

	double weight_H1toO_T[NUM_OF_OUT_NODE][NUM_OF_HID1_NODE];

	double net_H1[NUM_OF_HID1_NODE];
	double out_H1[NUM_OF_HID1_NODE];
	double del_H1[NUM_OF_HID1_NODE];
	double err_H1[NUM_OF_HID1_NODE];

	double net_O[NUM_OF_OUT_NODE];
	double out_O[NUM_OF_OUT_NODE];
	double del_O[NUM_OF_OUT_NODE];
	double err_O[NUM_OF_OUT_NODE];

	int errorCount = 0;

public:
	AI(string trainingRef, string testingRef);

	void setInNode(string dataType, int dataNum);
	void setNet(string location);
	void setOut(string location);
	void setError(string location, int dataNum);
	void setDelta(string location);
	void setWeight(string location);

	double getSigmoid(double net);
	double getSigmoidPrime(double net);
	void setTranspose(string location);

	void getError(int dataNum, int repeat);
	void getOut(int dataNum);
	int getErrorCount();
};