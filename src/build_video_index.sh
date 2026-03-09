#!/bin/bash


TARGET_DIR="../data/VideoCorpus/Subtitles"
export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"

python process_video_corpus.py
python -m pyserini.index.lucene \
       --collection JsonCollection \
       --input $TARGET_DIR \
       --index ../data/indexes/video_baseline_index \
       --generator DefaultLuceneDocumentGenerator \
       --threads 25 --storeDocvectors --storeRaw

echo "finished!"