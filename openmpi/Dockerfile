# Base image com OpenMPI
FROM ubuntu:latest

# Set working directory
WORKDIR /app

# Instalar dependências
RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin \
    libopenmpi-dev \
    libjson-c-dev \
    liblzma-dev \
    nlohmann-json3-dev \
    && apt-get clean

# Configurar variáveis de ambiente para permitir execução como root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Copiar o código-fonte para o container
COPY main_openmpi.cpp /app

# Compilar o código
RUN mpic++ -o main_openmpi main_openmpi.cpp -lz -ljson-c -lm

# Configuração padrão para o container
CMD ["mpirun", "-np", "4", "./main_openmpi"]
