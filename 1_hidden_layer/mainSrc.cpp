#include "back_propagation.h"

#define trainingRef "training.dat"
#define testingRef "testing.dat"

int main() {
	AI ai = AI(trainingRef, testingRef);

	cout << "----------------------------[ training ]----------------------------" << endl;
	for (int repeat = 0; repeat < COUNT; repeat++) {
		for (int dataNum = 0; dataNum < NUM_OF_DATA; dataNum++) {
			//데이터 입력
			ai.setInNode("training", dataNum);

			//은닉층1 NET 계산
			ai.setNet("hid1");

			//은닉층1 OUT 계산
			ai.setOut("hid1");

			//출력층 NET 계산
			ai.setNet("out");

			//출력층 OUT 계산
			ai.setOut("out");

			//출력층 ERROR 계산
			ai.setError("out", dataNum);

			//출력층 DELTA 계산
			ai.setDelta("out");

			//은닉층1 ERROR 계산
			ai.setError("hid1", dataNum);

			//은닉층1 DELTA 계산
			ai.setDelta("hid1");

			//은닉층1 -> 출력층 가중치 조정
			ai.setWeight("H1toO");

			//입력층 -> 은닉층1 가중치 조정
			ai.setWeight("ItoH1");

			//에러 출력
			ai.getError(dataNum, repeat);
		}
	}
	cout << endl;

	cout << "----------------------------[ testing ]----------------------------" << endl;
	for (int dataNum = 0; dataNum < NUM_OF_DATA; dataNum++) {
		//데이터 입력
		ai.setInNode("testing", dataNum);

		//은닉층1 NET 계산
		ai.setNet("hid1");

		//은닉층1 OUT 계산
		ai.setOut("hid1");

		//출력층 NET 계산
		ai.setNet("out");

		//출력층 OUT 계산
		ai.setOut("out");
		
		//OUT 출력
		ai.getOut(dataNum);
	}

	
	return 0;
}