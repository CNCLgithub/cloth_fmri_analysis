#!/bin/bash

mkdir -p data
cd data

###############################################
## Download raw BOLD signal data
##
## Participant data are hosted on OSF:
##   - Participants 1-12:  https://osf.io/t3h5q
##   - Participants 13-24: https://osf.io/2xuh7
###############################################
mkdir -p raw_data
cd raw_data
rawdata_urls=("https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67052ff06bd71af6a132d89b"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67053c44723b53d4c39fd1fe"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67053e6a69ab994654fcafb4"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67053e7a373726505332d7e9"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/670540584418f8cedcfca875"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67053d37c9864608245cd378"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67052cae64c20c629ffcac59"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67052d5c6bd71af6a132d753"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67053234263593ddec9fd2c0"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67053d0616e48bcac2a1fdb9"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/6705412abfe9a2dc9bfca99f"
              "https://files.osf.io/v1/resources/t3h5q/providers/osfstorage/67052d38d8044adfe2a2065a"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/6705442368e746fe09fcae43"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/67055880fc6993a0949fd003"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/670543078c5502a2875ccaad"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/67055869730ef601ae32df08"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/670549ce796bbb716d5cc8d9"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/6705491c71de658aba9fcefc"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/670558cb801480d0dc5cc8f1"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/670543b7a4779bb0345cc9d1"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/67054464bd4e2422b05cce12"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/670543c1095b82fd01a1fe49"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/6705582491df79b6ebfcae14"
              "https://files.osf.io/v1/resources/2xuh7/providers/osfstorage/6705588662b925d86b32dab0"
              )

for url in "${rawdata_urls[@]}"; do
  filename="${url##*/}.zip"
  echo "Downloading $filename..."
  curl -L "$url" -o "$filename"
  unzip -o "$filename" && rm "$filename"
done
cd ..


###############################################
## Download GLMsingle beta maps
##
## Data are hosted on OSF:
##   https://osf.io/kdmrg
###############################################
mkdir -p glmsingle
cd glmsingle
beta_urls=("https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2de5c8efddada32d2e1"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b29e9efb6d299b9fd95a"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b264222f842893fcb066"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2de599a2c6629a20a6e"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2e3e39fb2ca39a1fe19"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b264cd20257fb6a208b6"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b26458915db107fca8ab"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b29f8d2084479e32d6aa"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2a3f952a4fb66a209b2"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2ab8d2084479e32d6b0"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2ae636fe19a8032df7e"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2b1611b7d0ebb5ccc47"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2cc74c61d13685cd465"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2cd200f06cbdcfcadf3"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2d1eb881bb45c9fca10"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2f958915db107fca8fa"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2fa622e4b19565cd37a"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b25fd5ca6c5904a2076f"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b25f622e4b19565cd362"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b283636fe19a8032df79"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b27da48948054f5cd15f"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b2811df1f1d9fba1ff69"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b282b619ef68a5fcaf77"
          "https://files.osf.io/v1/resources/kdmrg/providers/osfstorage/6705b28358915db107fca8bf"
          )

for url in "${beta_urls[@]}"; do
  filename="${url##*/}.zip"
  echo "Downloading $filename..."
  curl -L "$url" -o "$filename"
  unzip -o "$filename" && rm "$filename"
done

## Download the aggregated file
beta_all_url="https://osf.io/download/6a478a7c1e6282be8ddaa31e/"
echo "Downloading beta all file (no unzip)..."
curl -L -OJ "$beta_all_url"


cd ..




###############################################
## Download the roi files
##
## Data are hosted on OSF:
##   https://osf.io/kdmrg
###############################################
mkdir -p towerLoc_subject_conc-False_len-0-603_space-MNI152Lin
cd towerLoc_subject_conc-False_len-0-603_space-MNI152Lin


beta_urls=(
  "https://osf.io/download/6a48a5503dacf2a23adaa297"
  "https://osf.io/download/6a48a59856d32eed98df9632"
  "https://osf.io/download/6a48a54e38188ebea7df96a6"
  "https://osf.io/download/6a48a59695dc8082b9df951f"
  "https://osf.io/download/6a48a5e3b0230b75c5daa2ac"
  "https://osf.io/download/6a48a5e3cc9b7cce3edaa1f2"
  "https://osf.io/download/6a48a62eb0230b75c5daa2b9"
  "https://osf.io/download/6a48a64ee5ae364e4f44f7a4"
  "https://osf.io/download/6a48a54d5846760be5df975c"
  "https://osf.io/download/6a48a5e25647bc4190daa2b0"
  "https://osf.io/download/6a48a5e3e5ae364e4f44f78e"
  "https://osf.io/download/6a48a59a65b93a7f2844f794"
  "https://osf.io/download/6a48a551e5ae364e4f44f772"
  "https://osf.io/download/6a48a59a911ec4c7bc44f772"
  "https://osf.io/download/6a48a62cef080e5ffddf94e9"
  "https://osf.io/download/6a48a55119fd3f043fdf953f"
  "https://osf.io/download/6a48a9448e1ab3b86ddf9700"
  "https://osf.io/download/6a48a626b0230b75c5daa2b7"
  "https://osf.io/download/6a48a921438a59092ddaa216"
  "https://osf.io/download/6a48a631cc9b7cce3edaa24c"
  "https://osf.io/download/6a48a62ecc9b7cce3edaa24a"
  "https://osf.io/download/6a48a64e95dc8082b9df9549"
  "https://osf.io/download/6a48a8ff8e1ab3b86ddf96e0"
  "https://osf.io/download/6a48a50865b93a7f2844f780"
  "https://osf.io/download/6a48a5063dacf2a23adaa293"
  "https://osf.io/download/6a48a5e270f7f9bc4b44f6db"
  "https://osf.io/download/6a48a59a19fd3f043fdf9562"
)

for url in "${beta_urls[@]}"; do
  echo "Downloading: $url"

  # download (keep OSF filename if possible)
  curl -L -O -J "$url"

  # get the downloaded file name (safe way)
  filename=$(ls -t | head -n 1)

  # check if it's tar.gz
  if [[ "$filename" == *.tar.gz ]]; then
    echo "Extracting: $filename"
    tar -xzf "$filename"

    echo "Removing archive: $filename"
    rm "$filename"
  else
    echo "Skipping extract (not tar.gz): $filename"
  fi

done


cd ..


###############################################
## Download cloth stimuli
##
## Data are hosted on OSF:
##   https://osf.io/kdmrg
###############################################
curl -L "https://osf.io/download/byazn/" -o cloth_stimuli.tar.gz && \
tar -xzf cloth_stimuli.tar.gz && \
rm cloth_stimuli.tar.gz && \
echo "Successfully downloaded and extracted cloth stimuli."


###############################################
## Download Baker shape-scrambled images and model results
##
## Data are hosted on OSF:
##   https://osf.io/kdmrg
###############################################

curl -L "https://osf.io/download/xntjc/" -o baker_test.tar.gz && \
tar -xzf baker_test.tar.gz && \
rm baker_test.tar.gz && \
echo "Successfully downloaded and extracted Baker shape-scrambled images and model results."


###############################################
## Download processed events
##
## Data are hosted on OSF:
##   https://osf.io/qk39p
###############################################
# URLs to download
processed_events_url=(
  "https://files.osf.io/v1/resources/qk39p/providers/osfstorage/680a6dd4f2285b471c568e13"
)

for url in "${processed_events_url[@]}"; do
  filename="${url##*/}.zip"
  echo "Downloading $filename..."
  curl -L "$url" -o "$filename"
  unzip -o "$filename" && rm "$filename"
done
rm -r __MACOSX



