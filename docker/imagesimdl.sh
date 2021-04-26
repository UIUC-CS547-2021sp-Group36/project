#!/bin/sh

LIST_URL="https://sites.google.com/site/imagesimilaritydata/download/query_and_triplets.txt?attredirects=0&d=1"

wget --progress=bar:force:noscroll -O QUERY_AND_TRIPLETS.TXT "${LIST_URL}" || exit 1
grep "http" QUERY_AND_TRIPLETS.TXT | sort | uniq > dload.txt || exit 1
mkdir -p images
cd images

echo "Beginning to download images"

while read url
do
    wget -q ${url}.jpg -O $(echo ${url} | sed 's/^.*tbn://g')
done < ../dload.txt

echo "Done downloading images"

cd ..
