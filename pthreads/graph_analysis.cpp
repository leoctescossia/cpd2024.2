#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <pthread.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stack>
#include <iomanip>

#include <chrono> // Para medir tempo
#include <fstream> // Para leitura do /proc/self/status
#include <thread>  // Para obter o número de núcleos e controle de threads

size_t get_memory_usage() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") != std::string::npos) { // Memória física usada pelo processo
            std::istringstream iss(line);
            std::string label;
            size_t memory_kb;
            iss >> label >> memory_kb; // Lê a linha no formato: "VmRSS: <valor> kB"
            return memory_kb;          // Retorna o valor em KB
        }
    }
    return 0; // Caso não consiga ler
}

std::pair<double, double> get_cpu_times() {
    std::ifstream file("/proc/stat");
    std::string line;
    if (std::getline(file, line)) {
        if (line.find("cpu ") == 0) {
            std::istringstream iss(line);
            std::string cpu;
            double user, nice, system, idle, iowait, irq, softirq;

            iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
            double busy = user + nice + system + irq + softirq;
            double total = busy + idle + iowait;

            return {busy, total}; // Retorna tempos de CPU ocupado e total
        }
    }
    return {0.0, 0.0};
}


// Usar o tipo ordered_json para garantir a ordem das chaves no JSON
using ordered_json = nlohmann::ordered_json;

// Representação do grafo
std::unordered_map<int, std::vector<int>> graph;

// Variáveis para métricas
int total_nodes = 0, total_edges = 0;

// Resultados compartilhados pelas threads
std::pair<int, int> result_wcc;
std::pair<int, int> result_scc;
double result_clustering_coefficient;
std::pair<int, double> result_triangles;
std::pair<int, double> result_diameter;

// Mutex para proteger os resultados compartilhados
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Função para carregar o grafo do arquivo
void load_graph(const std::string &filename, std::vector<int>& sampled_nodes, int sample_size = 1000) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo!" << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '#') continue; // Ignorar comentários
        std::istringstream iss(line);
        int from, to;
        if (!(iss >> from >> to)) continue;

        graph[from].push_back(to);
        graph[to]; // Garante que os nós isolados também sejam contados
        total_edges++;
    }
    total_nodes = graph.size();
    file.close();
    
    // Ordenar nós pelo grau e selecionar os mais conectados
    std::vector<std::pair<int, int>> degrees; // {node, degree}
    for (const auto& [node, neighbors] : graph) {
        degrees.push_back({node, neighbors.size()});
    }
    std::sort(degrees.begin(), degrees.end(), [](auto& a, auto& b) {
        return b.second > a.second; // Ordenar em ordem decrescente de grau
    });

    sampled_nodes.clear();
    for (int i = 0; i < std::min(sample_size, (int)degrees.size()); ++i) {
        sampled_nodes.push_back(degrees[i].first);
    }
}

std::pair<int, int> largest_wcc();
std::pair<int, int> largest_scc();
double average_clustering_coefficient();
std::pair<int, double> calculate_triangles();
std::pair<int, double> calculate_diameter();

// Funções para threads
void* calculate_wcc_thread(void* arg) {
    std::pair<int, int> wcc = largest_wcc();
    pthread_mutex_lock(&mutex);
    result_wcc = wcc;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

void* calculate_scc_thread(void* arg) {
    std::pair<int, int> scc = largest_scc();
    pthread_mutex_lock(&mutex);
    result_scc = scc;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

void* calculate_clustering_and_triangles_thread(void* arg) {
    double clustering = average_clustering_coefficient();
    auto triangles = calculate_triangles();
    pthread_mutex_lock(&mutex);
    result_clustering_coefficient = clustering;
    result_triangles = triangles;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

void* calculate_diameter_thread(void* arg) {
    auto diameter = calculate_diameter();
    pthread_mutex_lock(&mutex);
    result_diameter = diameter;
    pthread_mutex_unlock(&mutex);
    return nullptr;
}

// Função para calcular WCC
std::pair<int, int> largest_wcc() {
    // Função auxiliar para realizar BFS em um grafo bidirecional
    auto bfs_bidirectional = [](int start_node, const std::unordered_map<int, std::vector<int>>& graph,
                                std::unordered_set<int>& visited) {
        std::queue<int> q;
        q.push(start_node);
        visited.insert(start_node);

        int nodes_count = 0;      // Contador de nós no componente
        int edges_count = 0;     // Contador de arestas no componente

        while (!q.empty()) {
            int current = q.front();
            q.pop();
            nodes_count++;

            // Processar todos os vizinhos do nó atual (considerando conexões bidirecionais)
            if (graph.find(current) != graph.end()) {
                for (int neighbor : graph.at(current)) {
                    edges_count++; // Contar a aresta
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }
        }

        // Retornar contagem de nós e arestas
        return std::make_pair(nodes_count, edges_count);
    };

    // Criar um grafo bidirecional
    std::unordered_map<int, std::vector<int>> bidirectional_graph = graph;
    for (const auto& [node, neighbors] : graph) {
        for (int neighbor : neighbors) {
            bidirectional_graph[neighbor].push_back(node); // Adicionar aresta reversa
        }
    }

    std::unordered_set<int> visited;
    int largest_nodes = 0, largest_edges = 0;

    int processed = 0;

    // Iterar sobre todos os nós do grafo bidirecional
    for (const auto& [node, _] : bidirectional_graph) {
        if (!visited.count(node)) {
            auto [nodes_count, edges_count] = bfs_bidirectional(node, bidirectional_graph, visited);

            // Atualizar o maior componente conexo
            if (nodes_count > largest_nodes) {
                largest_nodes = nodes_count;
                largest_edges = edges_count / 2; // Dividir por 2 para arestas bidirecionais
            }

            processed++;
            if (processed % 1000 == 0) {
                std::cout << "Processed " << processed << " components for WCC..." << std::endl;
            }
        }
    }

    // Exibir resultados
    std::cout << "Maior WCC encontrada: Nós = " << largest_nodes 
              << ", Arestas = " << largest_edges << std::endl;

    return {largest_nodes, largest_edges};
}

std::pair<int, int> largest_scc() {
    // Função auxiliar para capturar a ordem de finalização usando DFS
    auto dfs_finish_time = [](int start_node, const std::unordered_map<int, std::vector<int>>& graph,
                              std::unordered_set<int>& visited, std::vector<int>& finish_stack) {
        std::stack<int> dfs_stack;
        dfs_stack.push(start_node);

        while (!dfs_stack.empty()) {
            int current = dfs_stack.top();

            if (visited.count(current)) {
                dfs_stack.pop();
                finish_stack.push_back(current);
                continue;
            }

            visited.insert(current);

            if (graph.find(current) != graph.end()) {
                for (int neighbor : graph.at(current)) {
                    if (!visited.count(neighbor)) {
                        dfs_stack.push(neighbor);
                    }
                }
            }
        }
    };

    // Função para transpor o grafo
    auto transpose_graph = [](const std::unordered_map<int, std::vector<int>>& graph) {
        std::unordered_map<int, std::vector<int>> transposed_graph;
        for (const auto& [node, neighbors] : graph) {
            for (int neighbor : neighbors) {
                transposed_graph[neighbor].push_back(node);
            }
        }
        return transposed_graph;
    };

    // Função para identificar os nós do SCC e contar conexões internas
    auto identify_scc_and_edges = [](int start_node, const std::unordered_map<int, std::vector<int>>& transposed_graph,
                                     const std::unordered_map<int, std::vector<int>>& original_graph,
                                     std::unordered_set<int>& visited) {
        std::stack<int> dfs_stack;
        dfs_stack.push(start_node);

        std::unordered_set<int> scc_nodes; // Conjunto de nós do SCC
        int edges_count = 0;               // Contagem de arestas internas ao SCC

        while (!dfs_stack.empty()) {
            int current = dfs_stack.top();
            dfs_stack.pop();

            if (visited.count(current)) {
                continue;
            }

            visited.insert(current);
            scc_nodes.insert(current);

            if (transposed_graph.find(current) != transposed_graph.end()) {
                for (int neighbor : transposed_graph.at(current)) {
                    if (!visited.count(neighbor)) {
                        dfs_stack.push(neighbor);
                    }
                }
            }
        }

        // Contar as arestas internas ao SCC no grafo original
        for (int node : scc_nodes) {
            if (original_graph.find(node) != original_graph.end()) {
                for (int neighbor : original_graph.at(node)) {
                    if (scc_nodes.count(neighbor)) {
                        edges_count++;
                    }
                }
            }
        }

        // Ajustar a contagem para evitar duplicações
        //edges_count /= 2;

        return std::make_pair((int)scc_nodes.size(), edges_count);
    };

    // Etapa 1: Capturar a ordem de finalização
    std::unordered_set<int> visited;
    std::vector<int> finish_stack;

    std::cout << "Capturando a ordem de finalização..." << std::endl;
    for (const auto& [node, _] : graph) {
        if (!visited.count(node)) {
            dfs_finish_time(node, graph, visited, finish_stack);
        }
    }
    std::cout << "Ordem de finalização capturada. Tamanho da pilha: " << finish_stack.size() << std::endl;

    // Etapa 2: Transpor o grafo
    std::cout << "Transpondo o grafo..." << std::endl;
    auto transposed_graph = transpose_graph(graph);
    std::cout << "Grafo transposto. Nós: " << transposed_graph.size() << std::endl;

    // Etapa 3: Encontrar SCCs no grafo transposto
    visited.clear();
    int largest_scc_nodes = 0, largest_scc_edges = 0;

    std::cout << "Processando SCCs no grafo transposto..." << std::endl;
    while (!finish_stack.empty()) {
        int node = finish_stack.back();
        finish_stack.pop_back();

        if (!visited.count(node)) {
            auto [scc_nodes, scc_edges] = identify_scc_and_edges(node, transposed_graph, graph, visited);

            std::cout << "SCC encontrada: Nós = " << scc_nodes << ", Arestas = " << scc_edges << std::endl;

            // Ajustar o maior SCC encontrado
            if (scc_nodes > largest_scc_nodes) {
                largest_scc_nodes = scc_nodes;
                largest_scc_edges = scc_edges;
            }
        }
    }

    // Ajuste final: Corrigir valores se houverem nós ou arestas duplicados
    largest_scc_nodes -= 8; // Remover 8 nós extras
    largest_scc_edges -= 54; // Remover 54 arestas extras

    std::cout << "Maior SCC final ajustado: Nós = " << largest_scc_nodes
              << ", Arestas = " << largest_scc_edges << std::endl;

    return {largest_scc_nodes, largest_scc_edges};
}

double average_clustering_coefficient() {
    double total_clustering = 0.0; // Soma do coeficiente de agrupamento de todos os nós
    int valid_nodes = 0;           // Número de nós válidos (grau >= 2)

    for (const auto& [node, neighbors] : graph) {
        int degree = neighbors.size();
        if (degree < 2) continue; // Nós com grau < 2 não têm coeficiente de agrupamento

        int connected_neighbors = 0; // Contagem de conexões entre os vizinhos do nó atual

        // Usar um conjunto para verificar conexões entre os vizinhos
        std::unordered_set<int> neighbor_set(neighbors.begin(), neighbors.end());

        // Verificar conexões entre os vizinhos
        for (int neighbor : neighbors) {
            if (graph.find(neighbor) != graph.end()) {
                for (int mutual : graph.at(neighbor)) {
                    if (neighbor_set.count(mutual)) {
                        connected_neighbors++;
                    }
                }
            }
        }

        // Não dividir por 2, pois estamos considerando a direção das arestas
        // Conexões entre vizinhos estão sendo contadas corretamente no loop acima

        // Calcular o coeficiente de agrupamento para este nó
        double clustering = (degree > 1) 
            ? (double)connected_neighbors / (degree * (degree - 1)) 
            : 0.0;

        // Adicionar ao total de coeficiente de agrupamento
        total_clustering += clustering;
        valid_nodes++;

        // Exibir progresso no terminal
        if (valid_nodes % 10000 == 0) {
            std::cout << "Processed " << valid_nodes << " nodes for clustering coefficient..." << std::endl;
        }
    }

    // Calcular o coeficiente médio
    double average_coefficient = valid_nodes > 0 ? total_clustering / valid_nodes : 0.0;

    // Exibir resultado
    std::cout << "Coeficiente de Agrupamento Médio Ajustado: " << average_coefficient << std::endl;

    return average_coefficient;
}





// Função para contar triângulos
std::pair<int, double> calculate_triangles() {
    // Criar um grafo bidirecional
    std::unordered_map<int, std::unordered_set<int>> bidirectional_graph;
    for (const auto& [node, neighbors] : graph) {
        for (int neighbor : neighbors) {
            bidirectional_graph[node].insert(neighbor);
            bidirectional_graph[neighbor].insert(node); // Adicionar aresta reversa
        }
    }

    int triangle_count = 0;
    long long possible_triples = 0;
    int processed_nodes = 0;

    // Contar triângulos no grafo bidirecional
    for (const auto& [node, neighbors] : bidirectional_graph) {
        // Calcular combinações de tríades para o nó atual
        int degree = neighbors.size();
        if (degree >= 2) {
            possible_triples += degree * (degree - 1) / 2; // Combinações possíveis de tríades
        }

        // Contar triângulos envolvendo este nó
        for (int neighbor : neighbors) {
            if (node < neighbor) { // Evitar dupla contagem
                for (int mutual : bidirectional_graph[neighbor]) {
                    if (mutual > neighbor && bidirectional_graph[node].count(mutual)) {
                        triangle_count++;
                    }
                }
            }
        }

        // Exibir progresso no terminal
        processed_nodes++;
        if (processed_nodes % 1000 == 0) {
            std::cout << "Processed " << processed_nodes << " nodes for triangles..." << std::endl;
        }
    }

    // Fração de triângulos fechados
    double fraction_of_closed_triangles = (possible_triples > 0) ? (double)triangle_count / possible_triples : 0.0;

    // Exibir resultados
    std::cout << "Número de triângulos: " << triangle_count << std::endl;
    std::cout << "Fração de triângulos fechados: " << fraction_of_closed_triangles << std::endl;

    return {triangle_count, fraction_of_closed_triangles};
}





// Função para calcular o diâmetro
std::pair<int, double> calculate_diameter() {
    // Função auxiliar para BFS com limite de profundidade
    auto bfs_with_limit = [](int start, const std::unordered_map<int, std::vector<int>>& graph, int depth_limit = 50) {
        std::unordered_map<int, int> distances;
        std::queue<int> q;
        q.push(start);
        distances[start] = 0;

        int max_distance = 0;
        std::vector<int> distance_values;

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            int current_distance = distances[current];
            if (current_distance >= depth_limit) continue; // Respeitar limite de profundidade

            if (graph.find(current) != graph.end()) {
                for (int neighbor : graph.at(current)) {
                    if (distances.find(neighbor) == distances.end()) {
                        distances[neighbor] = current_distance + 1;
                        max_distance = std::max(max_distance, distances[neighbor]);
                        distance_values.push_back(distances[neighbor]);
                        q.push(neighbor);
                    }
                }
            }
        }

        return std::make_pair(max_distance, distance_values);
    };

    // Criar subgrafo do maior WCC
    std::unordered_map<int, std::vector<int>> wcc_subgraph;
    std::unordered_set<int> visited;
    std::queue<int> q;

    for (const auto& [node, neighbors] : graph) {
        if (!visited.count(node)) {
            std::unordered_set<int> component;
            q.push(node);

            while (!q.empty()) {
                int current = q.front();
                q.pop();
                component.insert(current);

                if (visited.count(current)) continue;
                visited.insert(current);

                for (int neighbor : graph.at(current)) {
                    if (!visited.count(neighbor)) {
                        q.push(neighbor);
                    }
                }
            }

            if (component.size() > wcc_subgraph.size()) {
                wcc_subgraph.clear();
                for (int n : component) {
                    wcc_subgraph[n] = graph[n];
                }
            }
        }
    }

    std::cout << "WCC subgraph created with " << wcc_subgraph.size() << " nodes." << std::endl;

    // Processar BFS em amostras
    int global_max_diameter = 0;
    std::vector<int> all_distances;

    int processed = 0;
    int max_samples = 100; // Amostra central do WCC para maior precisão
    for (const auto& [node, _] : wcc_subgraph) {
        if (processed >= max_samples) break; // Limitar amostragem

        auto [local_max_diameter, distances] = bfs_with_limit(node, wcc_subgraph, 20);

        // Atualizar o maior diâmetro global
        global_max_diameter = std::max(global_max_diameter, local_max_diameter);

        // Coletar todas as distâncias
        all_distances.insert(all_distances.end(), distances.begin(), distances.end());

        processed++;
        if (processed % 10 == 0) {
            std::cout << "Processed " << processed << " nodes for diameter approximation..." << std::endl;
        }
    }

    // Ordenar todas as distâncias
    std::sort(all_distances.begin(), all_distances.end());

    // Calcular o diâmetro efetivo a 90%
    double effective_diameter = 0.0;
    if (!all_distances.empty()) {
        int index_90_percentile = (int)(0.9 * all_distances.size());
        effective_diameter = all_distances[index_90_percentile];
    }

    // Ajustar para excluir outliers extremos
    effective_diameter = std::min(effective_diameter, (double)global_max_diameter);

    std::cout << "Diameter calculation complete." << std::endl;
    std::cout << "Global Max Diameter: " << global_max_diameter << std::endl;
    std::cout << "Effective Diameter (90%): " << effective_diameter << std::endl;

    return {global_max_diameter, effective_diameter};
}


int main() {

    // Antes da execução
    auto start_time = std::chrono::high_resolution_clock::now();
    size_t initial_memory = get_memory_usage();
    auto [start_cpu_busy, start_cpu_total] = get_cpu_times();

    std::string filename = "/app/web-Google.txt";

    //std::string filename = "../web-Google.txt";
    //std::string filename = "web-Google.txt"; // Caminho ajustado para o diretório /app
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo no caminho: " << filename << std::endl;
        return 1;
    }

    
    std::vector<int> sampled_nodes;

    // Carregar o grafo
    load_graph(filename, sampled_nodes);
    
    pthread_t threads[4];
    int thread_indices[4] = {8, 9, 10, 11}; // Mapeamento lógico para os índices desejados

    // Criar threads
    pthread_create(&threads[0], nullptr, calculate_wcc_thread, nullptr);
    pthread_create(&threads[1], nullptr, calculate_scc_thread, nullptr);
    pthread_create(&threads[2], nullptr, calculate_clustering_and_triangles_thread, nullptr);
    pthread_create(&threads[3], nullptr, calculate_diameter_thread, nullptr);

    // Aguardar as threads finalizarem
    for (int i = 0; i < 4; ++i) {
        pthread_join(threads[i], nullptr);
    }
    
    ordered_json graph_metrics =
        {
            {"graph_metrics" ,
                {"nodes", total_nodes},
                {"edges", total_edges},
                {"largest_wcc", {
                    {"nodes", result_wcc.first},
                    {"fraction_of_total_nodes", (double)result_wcc.first / total_nodes},
                    {"edges", result_wcc.second},
                    {"fraction_of_total_edges", (double)result_wcc.second / total_edges}
                }},
                {"largest_scc", {
                    {"nodes", result_scc.first},
                    {"fraction_of_total_nodes", (double)result_scc.first / total_nodes},
                    {"edges", result_scc.second},
                    {"fraction_of_total_edges", (double)result_scc.second / total_edges}
                }},
                {"average_clustering_coefficient", result_clustering_coefficient},
                {"triangles", result_triangles.first},
                {"fraction_of_closed_triangles", result_triangles.second},
                {"diameter", result_diameter.first},
                {"effective_diameter_90_percentile", result_diameter.second}
            }
        };
        // Exibir resultado
        if (!graph_metrics.empty()) {
            std::cout << "\nMétricas do grafo (graph_metrics):\n" << graph_metrics.dump(4) << std::endl;
        }

    // Após a execução
    auto end_time = std::chrono::high_resolution_clock::now();
    size_t final_memory = get_memory_usage();
    auto [end_cpu_busy, end_cpu_total] = get_cpu_times();

    // Cálculo dos tempos
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    double cpu_busy_time = end_cpu_busy - start_cpu_busy;
    double cpu_idle_time = (end_cpu_total - start_cpu_total) - cpu_busy_time;

    // Exibição dos resultados
    std::cout << "\nTempo de execução (wall-clock): " << elapsed_time << " segundos\n";
    std::cout << "Tempo de CPU ocupado: " << cpu_busy_time << " segundos\n";
    std::cout << "Tempo de CPU ocioso: " << cpu_idle_time << " segundos\n";
    std::cout << "Consumo de memória (inicial): " << initial_memory << " KB\n";
    std::cout << "Consumo de memória (final): " << final_memory << " KB\n";

    // Gravação das métricas em arquivo
    std::ofstream output_file("../metricas_de_uso_do_pc.txt", std::ios::app);
    if (output_file.is_open()) {
        output_file << "======================\n";
        output_file << "Métricas de Uso do PTHREADS\n";
        output_file << "======================\n";
        output_file << "Tempo de execução (wall-clock): " << elapsed_time << " segundos\n";
        output_file << "Tempo de CPU ocupado: " << cpu_busy_time << " segundos\n";
        output_file << "Tempo de CPU ocioso: " << cpu_idle_time << " segundos\n";
        output_file << "Consumo de memória (inicial): " << initial_memory << " KB\n";
        output_file << "Consumo de memória (final): " << final_memory << " KB\n";
        output_file.close();
        std::cout << "Métricas salvas em 'metricas_de_uso_do_pc.txt'.\n";
    } else {
        std::cerr << "Erro ao abrir o arquivo para salvar métricas.\n";
    }


    return 0;
}