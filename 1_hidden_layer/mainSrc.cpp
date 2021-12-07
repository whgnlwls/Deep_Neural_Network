#include "back_propagation.h"

#define trainingRef "training.dat"
#define testingRef "testing.dat"

int main() {
	AI ai = AI(trainingRef, testingRef);

	cout << "----------------------------[ training ]----------------------------" << endl;
	for (int repeat = 0; repeat < COUNT; repeat++) {
		for (int dataNum = 0; dataNum < NUM_OF_DATA; dataNum++) {
			//������ �Է�
			ai.setInNode("training", dataNum);

			//������1 NET ���
			ai.setNet("hid1");

			//������1 OUT ���
			ai.setOut("hid1");

			//����� NET ���
			ai.setNet("out");

			//����� OUT ���
			ai.setOut("out");

			//����� ERROR ���
			ai.setError("out", dataNum);

			//����� DELTA ���
			ai.setDelta("out");

			//������1 ERROR ���
			ai.setError("hid1", dataNum);

			//������1 DELTA ���
			ai.setDelta("hid1");

			//������1 -> ����� ����ġ ����
			ai.setWeight("H1toO");

			//�Է��� -> ������1 ����ġ ����
			ai.setWeight("ItoH1");

			//���� ���
			ai.getError(dataNum, repeat);
		}
	}
	cout << endl;

	cout << "----------------------------[ testing ]----------------------------" << endl;
	for (int dataNum = 0; dataNum < NUM_OF_DATA; dataNum++) {
		//������ �Է�
		ai.setInNode("testing", dataNum);

		//������1 NET ���
		ai.setNet("hid1");

		//������1 OUT ���
		ai.setOut("hid1");

		//����� NET ���
		ai.setNet("out");

		//����� OUT ���
		ai.setOut("out");
		
		//OUT ���
		ai.getOut(dataNum);
	}

	
	return 0;
}