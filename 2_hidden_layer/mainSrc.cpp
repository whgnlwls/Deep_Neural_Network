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

			//������2 NET ���
			ai.setNet("hid2");
			
			//������2 OUT ���
			ai.setOut("hid2");

			//����� NET ���
			ai.setNet("out");

			//����� OUT ���
			ai.setOut("out");

			//����� ERROR ���
			ai.setError("out", dataNum);

			//����� DELTA ���
			ai.setDelta("out");

			//������2 ERROR ���
			ai.setError("hid2", dataNum);

			//������2 DELTA���
			ai.setDelta("hid2");

			//������1 ERROR ���
			ai.setError("hid1", dataNum);

			//������1 DELTA ���
			ai.setDelta("hid1");

			//������2 -> ����� ����ġ ����
			ai.setWeight("H2toO");

			//������1 -> ������2 ����ġ ����
			ai.setWeight("H1toH2");

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

		//������2 NET ���
		ai.setNet("hid2");

		//������2 OUT ���
		ai.setOut("hid2");

		//����� NET ���
		ai.setNet("out");

		//����� OUT ���
		ai.setOut("out");
		
		//OUT ���
		ai.getOut(dataNum);
	}

	
	return 0;
}