#!/bin/bash

#download and index pubmed baseline using lucene-based pyserini

URL="http://bionlp.nlm.nih.gov/pubmed_corpus_2025.jsonl.tar.gz"
ZIP_FILE=$(basename "$URL")
TARGET_DIR="../data/PubMedCorpus"

echo "Downloading file..."
wget -O "$ZIP_FILE" "$URL"

TEMP_DIR="temp_extraction_dir"
mkdir -p "$TEMP_DIR"

echo "Extracting tar.gz file..."
tar -xzf "$ZIP_FILE" -C "$TEMP_DIR"

EXTRACTED_SUBDIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -d "$EXTRACTED_SUBDIR" ]; then
    echo "Renaming/moving extracted folder to $TARGET_DIR..."
    rm -rf "$TARGET_DIR"
    mv "$EXTRACTED_SUBDIR" "$TARGET_DIR"
else
    echo "No folder found in archive; copying files directly to $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
    mv "$TEMP_DIR"/* "$TARGET_DIR"/
fi

echo "Cleaning up..."
rm -rf "$ZIP_FILE"
rm -rf "$TEMP_DIR"


export JAVA_HOME="$HOME/jdk/jdk-21.0.1+12"
export PATH="$JAVA_HOME/bin:$PATH"

/data/guptadk/anaconda3/envs/medgenvidqa/bin/python -m pyserini.index.lucene \
       --collection JsonCollection \
       --input $TARGET_DIR \
       --index ../data/indexes/pubmed_baseline_index \
       --generator DefaultLuceneDocumentGenerator \
       --threads 25 \
       --storePositions --storeDocvectors --storeRaw

echo "finished!"