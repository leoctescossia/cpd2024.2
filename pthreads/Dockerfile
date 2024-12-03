# Base image
FROM ubuntu:latest

# Set working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libjson-c-dev \
    liblzma-dev \
    nlohmann-json3-dev

# Copy necessary source code
COPY graph_analysis.cpp /app
#COPY web-Google.txt /app

# Compile the program
RUN g++ -o graph_analysis graph_analysis.cpp -lpthread -ljson-c -lm

# Set the default command to execute the compiled program
CMD ["./graph_analysis"]
