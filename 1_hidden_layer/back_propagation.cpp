#include "back_propagation.h"

AI::AI(string trainingRef, string testingRef) {
	//데이터 로드 시작
	ifstream dat_tr(trainingRef);
	if (dat_tr.is_open()) {
		double d_dat;
		for (int i = 0; i < NUM_OF_DATA; i++) {
			for (int j = 0; j < NUM_OF_IN_NODE; j++) {
				dat_tr >> d_dat;
				if (!isspace((int)d_dat)) {
					trainingDat[i][j] = d_dat;
				}
			}
		}
	}
	dat_tr.close();

	ifstream dat_te(testingRef);
	if (dat_te.is_open()) {
		double d_dat;
		for (int i = 0; i < NUM_OF_DATA; i++) {
			for (int j = 0; j < NUM_OF_IN_NODE; j++) {
				dat_te >> d_dat;
				if (!isspace((int)d_dat)) {
					testingDat[i][j] = d_dat;
				}
			}
		}
	}
	dat_te.close();
	//데이터 로드 끝

	//노드 초기화 시작
	for (int i = 0; i < NUM_OF_IN_NODE; i++) {
		inputDat[i] = 0;
	}
	for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
		net_H1[i] = 0;
		out_H1[i] = 0;
		err_H1[i] = 0;
		del_H1[i] = 0;
	}
	for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
		net_O[i] = 0;
		out_O[i] = 0;
		err_O[i] = 0;
		del_O[i] = 0;
	}
	//노드 초기화 끝

	//가중치 초기화 시작
	for (int i = 0; i < NUM_OF_IN_NODE; i++) {
		for (int j = 0; j < NUM_OF_HID1_NODE; j++) {
			weight_ItoH1[i][j] = (double)rand() / RAND_MAX;
		}
	}

	for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
		for (int j = 0; j < NUM_OF_OUT_NODE; j++) {
			weight_H1toO[i][j] = (double)rand() / RAND_MAX;
		}
	}
	//가중치 초기화 끝
}

//입력 노드 설정
void AI::setInNode(string dataType, int dataNum) {
	if (dataType == "training") {
		for (int i = 0; i < NUM_OF_IN_NODE; i++) {
			inputDat[i] = trainingDat[dataNum][i];
		}
	}
	else if (dataType == "testing") {
		for (int i = 0; i < NUM_OF_IN_NODE; i++) {
			inputDat[i] = testingDat[dataNum][i];
		}
	}
}

//NET 계산
void AI::setNet(string location) {
	double total;

	if (location == "hid1") {
		for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
			total = 0;
			for (int j = 0; j < NUM_OF_IN_NODE; j++) {
				total += inputDat[j] * weight_ItoH1[j][i];
			}
			net_H1[i] = total;
		}
	}
	else if (location == "out") {
		for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
			total = 0;
			for (int j = 0; j < NUM_OF_HID1_NODE; j++) {
				total += out_H1[j] * weight_H1toO[j][i];
			}
			net_O[i] = total;
		}
	}
}

//OUT 계산
void AI::setOut(string location) {
	if (location == "hid1") {
		for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
			out_H1[i] = getSigmoid(net_H1[i]);
		}
	}
	else if (location == "out") {
		for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
			out_O[i] = getSigmoid(net_O[i]);
		}
	}
}

//ERROR 계산
void AI::setError(string location, int dataNum) {
	if (location == "hid1") {
		double total;

		setTranspose("H1toO");
		for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
			total = 0;
			for (int j = 0; j < NUM_OF_OUT_NODE; j++) {
				total += weight_H1toO_T[i][j] * del_O[j];
			}
			err_H1[i] = total;
		}
	}
	else if (location == "out") {
		if (0 <= dataNum && dataNum < 25) {
			for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
				err_O[i] = cls1[i] - out_O[i];
			}
		}
		else if (25 <= dataNum && dataNum < 50) {
			for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
				err_O[i] = cls2[i] - out_O[i];
			}
		}
		else if (50 <= dataNum && dataNum < 75) {
			for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
				err_O[i] = cls3[i] - out_O[i];
			}
		}
	}
}

//DELTA 계산
void AI::setDelta(string location) {
	if (location == "hid1") {
		for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
			del_H1[i] = getSigmoidPrime(net_H1[i]) * err_H1[i];
		}
	}
	else if (location == "out") {
		for (int i = 0; i < NUM_OF_OUT_NODE; i++) {
			del_O[i] = getSigmoidPrime(net_O[i]) * err_O[i];
		}
	}
}

//가중치 조정
void AI::setWeight(string location) {
	double deltaWeight = 0;

	if (location == "ItoH1") {
		for (int i = 0; i < NUM_OF_IN_NODE; i++) {
			for (int j = 0; j < NUM_OF_HID1_NODE; j++) {
				deltaWeight = ALPHA * inputDat[i] * del_H1[j];
				weight_ItoH1[i][j] = weight_ItoH1[i][j] + deltaWeight;
			}
		}
	}
	else if (location == "H1toO") {
		for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
			for (int j = 0; j < NUM_OF_OUT_NODE; j++) {
				deltaWeight = ALPHA * out_H1[i] * del_O[j];
				weight_H1toO[i][j] = weight_H1toO[i][j] + deltaWeight;
			}
		}
	}
}

//시그모이드 함수
double AI::getSigmoid(double net) {
	double out;

	out = 1 / (1 + exp(-net));

	return out;
}

//시그모이드 도함수
double AI::getSigmoidPrime(double net) {
	double out;

	out = (1 / (1 + exp(-net))) * (1 - (1 / (1 + exp(-net))));

	return out;
}

//전치행렬
void AI::setTranspose(string location) {
	if (location == "H1toO") {
		for (int i = 0; i < NUM_OF_HID1_NODE; i++) {
			for (int j = 0; j < NUM_OF_OUT_NODE; j++) {
				weight_H1toO_T[j][i] = weight_H1toO[i][j];
			}
		}
	}
}

//ERROR 출력
void AI::getError(int dataNum, int repeat) {
	cout << "[training " << repeat * NUM_OF_DATA + dataNum + 1
		 << "]e1[" << err_O[0]
		 << "]\te2[" << err_O[1] 
		 << "]\te3[" << err_O[2]
		 << "]\te_t[" << err_O[0] + err_O[1] + err_O[2] << "]"
		 << endl;
}

//OUT 출력
void AI::getOut(int dataNum) {
	double max = out_O[0];
	int maxNode = 0;
	for (int i = 1; i < NUM_OF_OUT_NODE; i++) {
		if (max < out_O[i]) {
			max = out_O[i];
			maxNode = i;
		}
	}

	cout << "[testing " << dataNum + 1 
		 << "]\to1[" << out_O[0]
		 << "]\to2[" << out_O[1]
		 << "]\to3[" << out_O[2]
		 << "]\tcls[" << maxNode + 1 << "]"
		 << endl;
}