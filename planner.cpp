#include <iostream>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

using namespace std;
int ansf[510][510];
// 定义方向
const int dx[] = {-1, 1, 0, 0};  // 上, 下, 左, 右
const int dy[] = {0, 0, -1, 1};
const char dir_char[] = {'U', 'D', 'L', 'R'};  // 用于输出的字符

struct Node {
    int x, y,
        dir;  // 坐标和方向(0:上, 1:下, 2:左, 3:右), dir表示进入该节点的来自方向
    long long cost;                         // 到达该节点的总成本
    vector<pair<int, int>> path;            // 路径记录
    vector<pair<int, int>> built_stations;  // 已经建站的坐标

    Node(int x, int y, int dir, long long cost, vector<pair<int, int>> path,
         vector<pair<int, int>> built)
        : x(x), y(y), dir(dir), cost(cost), path(path), built_stations(built) {}

    // 重载小于运算符，用于优先队列，cost小的优先级高
    bool operator>(const Node& other) const { return cost > other.cost; }
};

int main() {
    int rows, cols;
    cin >> rows >> cols;

    vector<vector<int>> move_cost(rows, vector<int>(cols));
    vector<vector<int>> build_cost(rows, vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cin >> move_cost[i][j];
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cin >> build_cost[i][j];
        }
    }

    int start_x, start_y, end_x, end_y;
    cin >> start_x >> start_y >> end_x >> end_y;

    int turn_cost;
    cin >> turn_cost;

    // 检查坐标有效性
    if (start_x < 0 || start_x >= rows || start_y < 0 || start_y >= cols ||
        end_x < 0 || end_x >= rows || end_y < 0 || end_y >= cols) {
        cout << "Invalid coordinates." << endl;
        return 1;
    }

    // 初始化优先队列和 visited 数组
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    // visited[x][y][dir][built_station]
    // visited 数组记录状态 (x, y, dir, built_station) 是否被访问过
    // dir: 进入节点 (x, y) 的方向 (0:上, 1:下, 2:左, 3:右)
    // built_station:  表示是否*已经*在该节点(x,y)建站
    vector<vector<vector<vector<bool>>>> visited(
        rows, vector<vector<vector<bool>>>(
                  cols, vector<vector<bool>>(4, vector<bool>(2, false))));

    // 将四个初始方向的节点加入队列, 并且分别考虑建站和不建站的两种情况
    for (int i = 0; i < 4; ++i) {
        vector<pair<int, int>> initial_path = {{start_x, start_y}};
        vector<pair<int, int>> initial_built;

        // 初始不建站
        pq.push(Node(start_x, start_y, i, move_cost[start_x][start_y],
                     initial_path, initial_built));

        // 初始建站
        initial_built.push_back({start_x, start_y});
        pq.push(Node(start_x, start_y, i,
                     move_cost[start_x][start_y] + build_cost[start_x][start_y],
                     initial_path, initial_built));
    }

    Node best_node = Node(-1, -1, -1, numeric_limits<long long>::max(), {}, {});

    // Dijkstra 算法主循环
    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();

        int x = current.x;
        int y = current.y;
        int dir = current.dir;
        long long cost = current.cost;
        vector<pair<int, int>> current_path = current.path;

        if (x == end_x && y == end_y) {
            if (cost < best_node.cost) {
                best_node = current;
            }
            continue;  // 可以继续探索其他可能的建站方案，找到全局最优,
                       // 不能直接break
        }

        // 检查当前状态是否已访问. has_built表示 *当前节点* 是否 *已经*建站
        bool has_built = false;
        for (const auto& station : current.built_stations) {
            if (station.first == x && station.second == y) {
                has_built = true;
                break;
            }
        }

        if (visited[x][y][dir][has_built]) continue;
        visited[x][y][dir][has_built] = true;

        // 尝试向四个方向移动 (这里就包含了不建站的情况)
        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                // 检查下一个点是否在当前路径中出现过
                bool visited_in_path = false;
                for (const auto& p : current_path) {
                    if (p.first == nx && p.second == ny) {
                        visited_in_path = true;
                        break;
                    }
                }
                if (visited_in_path) continue;

                long long new_cost =
                    cost + move_cost[nx][ny] + (i == dir ? 0 : turn_cost);
                vector<pair<int, int>> next_path = current_path;
                next_path.push_back({nx, ny});

                // 不建站, 直接移动过去
                pq.push(Node(nx, ny, i, new_cost, next_path,
                             current.built_stations));

                // 尝试在新位置建站 (nx, ny)
                bool next_has_built = false;
                for (const auto& station : current.built_stations) {
                    if (station.first == nx && station.second == ny) {
                        next_has_built = true;
                        break;
                    }
                }
                if (!next_has_built) {
                    vector<pair<int, int>> next_built_stations =
                        current.built_stations;
                    next_built_stations.push_back({nx, ny});
                    long long new_cost_build = new_cost + build_cost[nx][ny];
                    pq.push(Node(nx, ny, i, new_cost_build, next_path,
                                 next_built_stations));
                }
            }
        }
    }

    // 输出结果
    if (best_node.cost == numeric_limits<long long>::max()) {
        cout << "No path found." << endl;
    } else {
        cout << "Minimum cost: " << best_node.cost << endl;
        cout << "Path: ";
        for (size_t i = 0; i < best_node.path.size(); ++i) {
            cout << "(" << best_node.path[i].first << ", "
                 << best_node.path[i].second << ")";
            ansf[best_node.path[i].first][best_node.path[i].second] = 1;
            if (i < best_node.path.size() - 1) {
                cout << " -> ";
            }
        }
        cout << endl;

        cout << "Build Station: " << endl;
        if (best_node.built_stations.empty()) {
            cout << "No station built." << endl;
        } else {
            for (size_t i = 0; i < best_node.built_stations.size(); ++i) {
                cout << "(" << best_node.built_stations[i].first << ", "
                     << best_node.built_stations[i].second << ")" << endl;
                ansf[best_node.built_stations[i].first]
                    [best_node.built_stations[i].second] = 2;
            }
        }
    }
    return 0;
}
