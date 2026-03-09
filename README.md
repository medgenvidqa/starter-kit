# MedGenVidQA 2026 Starter Kit


Please install [Anaconda](https://www.anaconda.com/distribution/) to create a conda environment as follows:
```shell script
# preparing environment
conda create -n medgenvidqa2026 python=3.10
conda activate medgenvidqa2026
pip install -r requirements.txt
```


## Install Java Dependency
```shell script
wget https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.1+12/OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz
mkdir -p $HOME/jdk
tar -xzf OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz -C $HOME/jdk
export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"
```

## Install Pyserini Dependency
```shell script
conda install -c pytorch faiss-cpu -y
```




## Download HowTo100M caption which contains majority of the YouTube videos subtitles
```shell script
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/raw_caption.zip
```
Place the JSON file in ```data/HowTo100M```

## Download the TestDataset and VideoCorpus shared via CodaBench and place them in
```data/TestDataset``` and ```data/VideoCorpus``` respectively.


## Download Llama2 
Before downloading, you need to agree to Meta's license terms by visiting here: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

You may need to fill out the form to agree to the license terms. Once your request approved, run the following:


```shell script
  pip install huggingface_hub
  huggingface-cli login
  huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir llama-2-7b-chat-hf
```

## Download the fine-tuned adapter from [here](https://drive.google.com/drive/folders/1kydixzQ1ffkVK3dMsAKH7rl4cj7hJDCB?usp=sharing) and place them in ```data/FineTunedAdapter``` [Task B only]

Change directory to 
```
cd src/
```
## Download and index pubmed baseline
```shell script
./build_pubmed_index.sh
```
It will index 28,372,706 PubMed documents.


## Process and index video subtitles
```shell script
./build_video_index.sh
```


## Run Task A Baseline
```
 python task_a_baseline.py
```
The submission ready files (3 baseline approaches) will be saved in ```data/BaselineResults``` directory. 

## Run Task B Baseline
```
 python task_b_baseline.py
```

## Run Task C Baseline
```
 python task_c_baseline.py
