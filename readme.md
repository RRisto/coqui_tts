## Estonian TTS - Coqui model

Model files are taken from here: https://mygit.top/release/74967647

To use:

- build docker: `bash build_docker.sh`
  - run docker: `bash run_docker.sh`

      - if you want to use grpc protocol:
    ```
     docker run --rm \
        --platform=linux/amd64 \
        -p 9000:9000 \
        -p 8265:8265 \
        --shm-size=1g \
        --ulimit nofile=65536:65536 \
        -e RAY_USAGE_STATS_ENABLED=0 \
        -e PORT=9000 \
        -e SERVER_TYPE=grpc \
        -v "$PWD/model":/app/model:ro \
        "${IMAGE_NAME}:${IMAGE_TAG}"
    ```
     - if you want to use websocket:
    ```
     docker run --rm \
        --platform=linux/amd64 \
        -p 8080:8080 \
        -p 8265:8265 \
        --shm-size=1g \
        --ulimit nofile=65536:65536 \
        -e RAY_USAGE_STATS_ENABLED=0 \
        -e PORT=8080 \
        -e SERVER_TYPE=websocket \
        -v "$PWD/model":/app/model:ro \
        "${IMAGE_NAME}:${IMAGE_TAG}"
    ```
    
    - run client `jupyter notebook` and run `test_coqui.ipynb` notebook
    - grpc assumes you have `tts_pb2.py` and `tts_pb2_grpc.py`
    
      - you can generate them `python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. tts.proto` 
      (assumes you have same `grpcio-tools` version as in docker)