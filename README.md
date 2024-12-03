# cpd2024.2
# Trabalho de Implementação - Computação Paralela e Distribuída

Este projeto contém implementações paralelas e distribuídas para análise de grafos usando PThreads, OpenMP e OpenMPI.

## Instruções

### Requisitos
- Podman
- Docker Compose
- zlib1g-dev
- nlohmann-json3-dev
- openmpi-bin
- libopenmpi-dev

### Como Executar

1. **Construir os contêineres**:
   ```bash
   podman-compose build
   ```

2. **Iniciar os serviços dos contêineres**:
   ```bash
   podman-compose up
   ```

### pthreads

```bash
g++ -o main_pthreads main_pthreads.cpp -lpthread -ljson-c -lm
```

### openMP

```bash
g++ -o main_openmp main_openmp.cpp -fopenmp -ljson-c -lm
```

### openMPI

```
mpic++ -o main_openmpi main_openmpi.cpp -lz -ljson-c -lm
```

```
chmod +x main_openmpi
```

```
mpirun -np 4 ./main_openmpi
```

# Agradecimentos

Gostaria de expressar minha sincera gratidão ao grupo de **Yuri e Gabriel** pela a colaboração e ajuda nessa longa jornada. Sem a colaboração deles, esse trabalho não seria concluído.
