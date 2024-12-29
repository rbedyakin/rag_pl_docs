# RAG on Pytorch Lightning docs
Local RAG with Ollama, LlamaIndex, Typesense, LLama3.2 (3b) on Pytorch Lightning docs.

## Install
```
python -m pip install -r requirements.txt
```

## Usage

### Load Data
```
mkdir data
wget https://github.com/Lightning-AI/lightning/archive/refs/heads/release/stable.zip -O "data/stable.zip"
unzip data/stable.zip -d data/
``` 

### Run Typesense
```
docker pull typesense/typesense

export TYPESENSE_API_KEY=xyz

mkdir $(pwd)/typesense-data

docker run -p 8108:8108 -v$(pwd)/typesense-data:/data typesense/typesense:27.1 \
  --data-dir /data --api-key=$TYPESENSE_API_KEY --enable-cors
```

### Run main.py to get answers
```
python main.py
```