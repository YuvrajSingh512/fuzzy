#include <iostream>
using namespace std;

void toh(int n, char from, char to, char via){
    if(n>0){
        toh(n-1,from,via,to);
        cout << "move disk no. " << n <<" from "<< from << " to "<< to << endl;
        toh(n-1,via,to,from);
    }
}

int main() {
	// your code goes here
	int n;
	cin >> n;
	toh(n,'A','C','B');
	return 0;
}
