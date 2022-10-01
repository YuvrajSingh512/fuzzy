#include <iostream>
using namespace std;

void printpascal(int n){
    for(int i=1;i<=n;i++){
        int c = 1;
        for(int j=1;j<=i;j++){
            cout << c << " ";
            c = c* (i-j)/j;
        }
        cout << endl;
    }
}

int main() {
	// your code goes here
	int n;
	cin >> n;
	printpascal(n);
	return 0;
}
