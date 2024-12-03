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
COPY grafomp.cpp /app
#COPY web-Google.txt /app

# Compile the program
RUN g++ -o grafomp grafomp.cpp -fopenmp -ljson-c -lm



# Set the default command to execute the compiled program
CMD ["./grafomp"]
