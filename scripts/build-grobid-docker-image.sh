#!/bin/bash

set -e

CLOUD_MODELS_PATH=${1:-$CLOUD_MODELS_PATH}
SOURCE_GROBID_IMAGE=${2:-$SOURCE_GROBID_IMAGE}
OUTPUT_GROBID_IMAGE=${3:-$OUTPUT_GROBID_IMAGE}
GCP_PROJECT=${4:-$GCP_PROJECT}

if [ -z "${CLOUD_MODELS_PATH}" ]; then
    echo "Error: CLOUD_MODELS_PATH required"
    exit 1
fi

if [ -z "${SOURCE_GROBID_IMAGE}" ]; then
    echo "Error: SOURCE_GROBID_IMAGE required"
    exit 1
fi

if [ -z "${OUTPUT_GROBID_IMAGE}" ]; then
    echo "Error: OUTPUT_GROBID_IMAGE required"
    exit 1
fi

echo "CLOUD_MODELS_PATH=${CLOUD_MODELS_PATH}"
echo "SOURCE_GROBID_IMAGE=${SOURCE_GROBID_IMAGE}"
echo "OUTPUT_GROBID_IMAGE=${OUTPUT_GROBID_IMAGE}"
echo "GCP_PROJECT=${GCP_PROJECT}"

temp_build_dir=/tmp/grobid-trained-model-build
rm -rf "${temp_build_dir}"
mkdir -p "${temp_build_dir}"
cp docker/grobid-with-trained-model/* "${temp_build_dir}"
gsutil ls -l "${CLOUD_MODELS_PATH}/**"
mkdir -p "${temp_build_dir}/models/header"
gsutil cat "${CLOUD_MODELS_PATH}/header/model.wapiti.gz" > "${temp_build_dir}/models/header/model.wapiti.gz"
find "${temp_build_dir}"

if [ -z "${GCP_PROJECT}" ]; then
    echo "build local image: $OUTPUT_GROBID_IMAGE"
    docker build \
        --build-arg "base_image=${SOURCE_GROBID_IMAGE}" \
        --tag ${OUTPUT_GROBID_IMAGE} \
        "${temp_build_dir}"
else
    echo "build image using gcloud build: $OUTPUT_GROBID_IMAGE"
    gcloud builds submit --project "${GCP_PROJECT}" \
        --config docker/grobid-with-trained-model/config.yaml \
        --substitutions "_BASE_IMAGE=${SOURCE_GROBID_IMAGE},_IMAGE=${OUTPUT_GROBID_IMAGE}" \
        "${temp_build_dir}"
fi
