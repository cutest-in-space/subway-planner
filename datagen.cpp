#include <bits/stdc++.h>

using namespace std;
mt19937 rnd(time(0));
int n, m, turncost, stx, sty, edx, edy;
int main() {
    cin >> n >> m >> stx >> sty >> edx >> edy >> turncost;
    cout << n << " " << m << "\n";
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cout << rnd() % 2000 << " ";
        }
        cout << "\n";
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            cout << int(rnd() % 4000) - 400 << " ";
        }
        cout << "\n";
    }
    cout << stx << " " << sty << " " << edx << " " << edy << "\n" << turncost;
    return 0;
}
