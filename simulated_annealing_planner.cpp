#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <set>
#include <sstream>

using namespace std;

// 全局变量：地图尺寸、移动成本、建站成本、起终点坐标及转向代价
int rows, cols;
vector<vector<int>> move_cost;
vector<vector<int>> build_cost;
int start_x, start_y, end_x, end_y;
int turn_cost;

// 定义解的表示：路径（连续的坐标）和每个节点是否建站的标志
struct Solution {
    vector<pair<int, int>> path;
    vector<bool> built; // 与 path 对应，true 表示该节点建站
};

// 成本函数：计算一条解的总成本
long long calc_cost(const Solution &sol) {
    long long cost = 0;
    // 移动成本：路径上每个节点的移动成本都要累加
    for (const auto &p : sol.path) {
        cost += move_cost[p.first][p.second];
    }
    // 建站成本
    for (size_t i = 0; i < sol.path.size(); i++) {
        if (sol.built[i])
            cost += build_cost[sol.path[i].first][sol.path[i].second];
    }
    // 转向代价：对连续两步之间方向变化（第一步没有转向代价）
    if (sol.path.size() >= 2) {
        int dx_prev = sol.path[1].first - sol.path[0].first;
        int dy_prev = sol.path[1].second - sol.path[0].second;
        for (size_t i = 2; i < sol.path.size(); i++) {
            int dx_cur = sol.path[i].first - sol.path[i - 1].first;
            int dy_cur = sol.path[i].second - sol.path[i - 1].second;
            if (dx_cur != dx_prev || dy_cur != dy_prev) {
                cost += turn_cost;
            }
            dx_prev = dx_cur;
            dy_prev = dy_cur;
        }
    }
    return cost;
}

// 辅助函数：判断路径是否简单（没有重复节点）
bool is_simple(const vector<pair<int, int>> &path) {
    set<pair<int, int>> seen;
    for (const auto &p : path) {
        if (seen.count(p))
            return false;
        seen.insert(p);
    }
    return true;
}

// 生成两点之间的曼哈顿路径（随机打乱必经的步序），保证从 s 到 t 的一个可行路径
vector<pair<int, int>> generate_manhattan_path(pair<int, int> s, pair<int, int> t) {
    vector<pair<int, int>> path;
    path.push_back(s);
    int dx = t.first - s.first;
    int dy = t.second - s.second;
    // 用数字表示移动方向：0：上, 1：下, 2：左, 3：右
    vector<int> moves;
    if (dx > 0) {
        for (int i = 0; i < dx; i++) moves.push_back(1);
    } else {
        for (int i = 0; i < -dx; i++) moves.push_back(0);
    }
    if (dy > 0) {
        for (int i = 0; i < dy; i++) moves.push_back(3);
    } else {
        for (int i = 0; i < -dy; i++) moves.push_back(2);
    }
    // 随机打乱移动序列
    random_shuffle(moves.begin(), moves.end());
    int cur_x = s.first, cur_y = s.second;
    for (int move : moves) {
        if (move == 0) cur_x--;      // 上
        else if (move == 1) cur_x++; // 下
        else if (move == 2) cur_y--; // 左
        else if (move == 3) cur_y++; // 右
        path.push_back({cur_x, cur_y});
    }
    return path;
}

// 邻域操作：从当前解产生一个新解
Solution neighbor(const Solution &sol) {
    Solution newSol = sol;
    int op = rand() % 2; // 选择操作：0—翻转某个节点的建站决策；1—修改路径结构
    if (op == 0) {
        // 随机选择一个非起点和终点节点翻转建站标记
        if (newSol.path.size() > 2) {
            int idx = 1 + rand() % (newSol.path.size() - 2);
            newSol.built[idx] = !newSol.built[idx];
        }
    } else {
        // 修改路径结构：随机选择路径中的两个位置 i 和 j (1 <= i < j <= n-2)，重新生成这段子路径
        if (newSol.path.size() > 3) { // 保证有中间段可修改
            int i = 1 + rand() % (newSol.path.size() - 2);
            int j = i + 1 + rand() % (newSol.path.size() - i - 1);
            pair<int, int> s = newSol.path[i];
            pair<int, int> t = newSol.path[j];
            vector<pair<int, int>> newSegment = generate_manhattan_path(s, t);
            // 构造候选路径：保留 [0, i] 段，接上新生成的子路径（去掉重复的起点和终点），再接上 [j, end] 段
            vector<pair<int, int>> candidate;
            vector<bool> candidate_built;
            for (int k = 0; k <= i; k++) {
                candidate.push_back(newSol.path[k]);
                candidate_built.push_back(newSol.built[k]);
            }
            for (size_t k = 1; k + 1 < newSegment.size(); k++) {
                candidate.push_back(newSegment[k]);
                candidate_built.push_back(false); // 新生成的点默认不建站
            }
            for (size_t k = j; k < newSol.path.size(); k++) {
                candidate.push_back(newSol.path[k]);
                candidate_built.push_back(newSol.built[k]);
            }
            // 若候选路径依然简单，则采用该修改；否则放弃此次操作，保持原解
            if (is_simple(candidate)) {
                newSol.path = candidate;
                newSol.built = candidate_built;
            }
        }
    }
    return newSol;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 读入地图尺寸、移动成本和建站成本
    cin >> rows >> cols;
    move_cost.resize(rows, vector<int>(cols));
    build_cost.resize(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            cin >> move_cost[i][j];
        }
    }
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            cin >> build_cost[i][j];
        }
    }
    // 读入起点、终点坐标和转向代价
    cin >> start_x >> start_y >> end_x >> end_y;
    cin >> turn_cost;

    // 读入模拟退火参数（如果提供）
    double T = 1000.0;      // 默认初始温度
    double alpha = 0.995;   // 默认降温速率
    int max_iter = 100000;  // 默认最大迭代次数
    
    // 尝试读取自定义参数
    string line;
    if (getline(cin, line) && !line.empty()) {
        // 跳过可能的空行
        if (line.empty() && getline(cin, line)) {}
        
        // 如果有参数行，解析参数
        if (!line.empty()) {
            istringstream iss(line);
            if (!(iss >> T >> alpha >> max_iter)) {
                // 如果解析失败，使用默认值
                T = 1000.0;
                alpha = 0.995;
                max_iter = 100000;
            }
        }
    }

    // 检查坐标有效性
    if (start_x < 0 || start_x >= rows || start_y < 0 || start_y >= cols ||
        end_x < 0 || end_x >= rows || end_y < 0 || end_y >= cols) {
        cout << "Invalid coordinates." << endl;
        return 1;
    }

    srand(time(NULL));

    // 初始解：采用曼哈顿路径（从起点到终点）
    Solution current;
    current.path = generate_manhattan_path({start_x, start_y}, {end_x, end_y});
    
    // 优化初始解：起点和终点必须建站，其他点有30%的概率建站
    current.built.resize(current.path.size(), false);
    current.built[0] = true;  // 起点建站
    current.built[current.path.size() - 1] = true;  // 终点建站
    
    // 其他点有30%的概率建站
    for (size_t i = 1; i < current.path.size() - 1; i++) {
        if ((double)rand() / RAND_MAX < 0.3) {
            current.built[i] = true;
        }
    }
    
    // 确保至少有4个站点（包括起点和终点）
    int station_count = 0;
    for (bool b : current.built) {
        if (b) station_count++;
    }
    
    if (station_count < 4) {
        // 随机选择额外的点建站
        vector<int> non_station_indices;
        for (size_t i = 1; i < current.path.size() - 1; i++) {
            if (!current.built[i]) {
                non_station_indices.push_back(i);
            }
        }
        
        random_shuffle(non_station_indices.begin(), non_station_indices.end());
        
        int to_add = min(4 - station_count, (int)non_station_indices.size());
        for (int i = 0; i < to_add; i++) {
            current.built[non_station_indices[i]] = true;
        }
    }
    
    long long current_cost = calc_cost(current);

    Solution best = current;
    long long best_cost = current_cost;

    // 模拟退火参数设置 - 使用从输入读取的参数
    double T_min = 1e-3;  // 温度下限
    int iteration = 0;

    // 模拟退火主循环
    while (T > T_min && iteration < max_iter) {
        Solution candidate = neighbor(current);
        
        // 确保起点和终点总是建站
        candidate.built[0] = true;
        candidate.built[candidate.path.size() - 1] = true;
        
        long long candidate_cost = calc_cost(candidate);
        long long delta = candidate_cost - current_cost;
        
        // 如果新解更优或者以一定概率接受劣解，则更新当前解
        if (delta < 0 || exp(-delta / T) > (double)rand() / RAND_MAX) {
            current = candidate;
            current_cost = candidate_cost;
            if (current_cost < best_cost) {
                best = current;
                best_cost = current_cost;
            }
        }
        
        T *= alpha;
        iteration++;
        
        // 每1000次迭代检查一次站点数量，如果太少则增加
        if (iteration % 1000 == 0) {
            int station_count = 0;
            for (bool b : current.built) {
                if (b) station_count++;
            }
            
            // 如果站点太少（少于路径长度的20%，且少于4个），增加建站概率
            if (station_count < max(4, (int)(current.path.size() * 0.2))) {
                vector<int> non_station_indices;
                for (size_t i = 1; i < current.path.size() - 1; i++) {
                    if (!current.built[i]) {
                        non_station_indices.push_back(i);
                    }
                }
                
                if (!non_station_indices.empty()) {
                    random_shuffle(non_station_indices.begin(), non_station_indices.end());
                    int to_add = min(2, (int)non_station_indices.size());  // 每次最多添加2个站点
                    for (int i = 0; i < to_add; i++) {
                        current.built[non_station_indices[i]] = true;
                    }
                    current_cost = calc_cost(current);
                }
            }
        }
    }

    // 输出结果
    cout << "Minimum cost: " << best_cost << "\n";
    cout << "Path: ";
    for (size_t i = 0; i < best.path.size(); i++){
        cout << "(" << best.path[i].first << ", " << best.path[i].second << ")";
        if (i != best.path.size() - 1)
            cout << " -> ";
    }
    cout << "\n";

    cout << "Build Station: " << "\n";
    bool anyBuilt = false;
    for (size_t i = 0; i < best.path.size(); i++){
        if (best.built[i]){
            cout << "(" << best.path[i].first << ", " << best.path[i].second << ")\n";
            anyBuilt = true;
        }
    }
    if (!anyBuilt)
        cout << "No station built." << "\n";

    return 0;
}
