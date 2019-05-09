DOCKER_COMPOSE_DEV = docker-compose
DOCKER_COMPOSE_CI = docker-compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)

RUN_GROBID_TRAINER = $(DOCKER_COMPOSE) run --rm grobid-trainer

RUN_TOOLS = $(DOCKER_COMPOSE) run --rm tools
RUN_TOOLS_DEV = $(DOCKER_COMPOSE) run --rm tools-dev

SAMPLE_DATA_PATH = gs://sciencebeam-samples/pmc-sample-1943-cc-by-subset
SAMPLE_DATA_NAME = pmc-sample-1943-cc-by-subset
SAMPLE_FILE_LIST = file-list.tsv

# Note: this could do with refactory to use a file list
HEADER_XML_TO_TARGET_XML_FILENAME_PATTERN = /(.*).header.tei.xml/\1.xml/

PDF_DATA_DIR = /data/pdf
RAW_TRAINING_DATA_DIR = /data/raw-training-data
DATASET_DIR = /data/dataset
XML_DATA_DIR = $(DATASET_DIR)/xml
AUTO_ANNOTATED_DATASET_DIR = /data/dataset-auto-annotated


GCP_PROJECT = elife-ml
GROBID_TAG = 0.5.4
GROBID_REPO = lfoppiano/grobid
GROBID_TRAINED_MODEL_IMAGE_REPO = gcr.io/$(GCP_PROJECT)/grobid-trained-model--dev

TRAIN_LIMIT = 5

SAMPLE_CLOUD_XML_PATH = ${SAMPLE_DATA_PATH}
SAMPLE_CLOUD_BASE_OUTPUT_PATH = $(SAMPLE_DATA_PATH)-models/dev/$(GROBID_TAG)/limit-$(TRAIN_LIMIT)
SAMPLE_CLOUD_DATASET_PATH = $(SAMPLE_CLOUD_BASE_OUTPUT_PATH)/grobid-dataset
SAMPLE_CLOUD_MODELS_PATH = $(SAMPLE_CLOUD_BASE_OUTPUT_PATH)/grobid-models


dev-venv:
	rm -rf venv || true

	virtualenv -p python2.7 venv

	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements.dev.txt
	export SCIENCEBEAM_GYM_NO_APT=1
	venv/bin/pip install -r requirements.links.txt


example-data-processing-end-to-end: \
	get-example-data \
	generate-grobid-training-data \
	auto-annotate-header \
	copy-auto-annotate-header-training-data-to-tei \
	train-header-model


example-data-processing-end-to-end-cloud: \
	get-example-data-cloud \
	download-dataset-pdf \
	generate-grobid-training-data \
	upload-dataset \
	auto-annotate-header-cloud \
	copy-auto-annotate-header-training-data-to-tei-cloud \
	train-and-upload-header-model-cloud


build:
	$(DOCKER_COMPOSE) build tools


build-dev:
	$(DOCKER_COMPOSE) build tools-dev-base-image tools-dev


build-docker:
	$(DOCKER_COMPOSE) build tools tools-docker


generate-raw-grobid-training-data:
	$(RUN_GROBID_TRAINER) generate-raw-grobid-training-data.sh \
		"${PDF_DATA_DIR}" \
		"$(RAW_TRAINING_DATA_DIR)"


copy-raw-training-data-to-file-structure:
	$(RUN_GROBID_TRAINER) copy-raw-training-data-to-file-structure.sh \
		"${RAW_TRAINING_DATA_DIR}" \
		 "$(DATASET_DIR)"


generate-grobid-training-data:
	$(RUN_GROBID_TRAINER) generate-grobid-training-data.sh \
		"${PDF_DATA_DIR}" \
		"$(DATASET_DIR)"


download-dataset-pdf:
	$(RUN_GROBID_TRAINER) download-dataset-pdf.sh "$(SAMPLE_CLOUD_DATASET_PATH)" "$(PDF_DATA_DIR)"


upload-dataset:
	$(RUN_GROBID_TRAINER) upload-dataset.sh "$(DATASET_DIR)" "$(SAMPLE_CLOUD_DATASET_PATH)"


train-header-model:
	@echo to train with default dataset add '--use-default-dataset',
	@echo e.g. 'make TRAIN_ARGS="--use-default-dataset" train-header-model'
	@echo you could also train using remote dataset,
	@echo e.g. 'make DATASET_DIR=gs://.../grobid-training-data train-header-model'
	$(RUN_GROBID_TRAINER) train-header-model.sh \
		--dataset "$(DATASET_DIR)" \
		$(TRAIN_ARGS)


upload-header-model:
	$(RUN_GROBID_TRAINER) upload-header-model.sh "$(SAMPLE_CLOUD_MODELS_PATH)"


train-and-upload-header-model:
	$(MAKE) \
		TRAIN_ARGS="--cloud-models-path $(SAMPLE_CLOUD_MODELS_PATH) $(TRAIN_ARGS)" \
		train-header-model


train-and-upload-header-model-cloud:
	$(MAKE) DATASET_DIR="$(SAMPLE_CLOUD_DATASET_PATH)" \
		TRAIN_ARGS="--cloud-models-path $(SAMPLE_CLOUD_MODELS_PATH) $(TRAIN_ARGS)" \
		train-header-model


grobid-shell:
	$(RUN_GROBID_TRAINER) bash


get-example-data: build
	$(RUN_TOOLS) python -m sciencebeam_trainer_grobid_tools.download_source_files \
		--document-file-list "$(SAMPLE_DATA_PATH)/$(SAMPLE_FILE_LIST)" \
		--document-file-column "source_url" \
		--target-file-list "$(SAMPLE_DATA_PATH)/$(SAMPLE_FILE_LIST)" \
		--target-file-column "xml_url" \
		--document-output-path "$(PDF_DATA_DIR)" \
		--document-output-filename-pattern "{name}.pdf" \
		--target-output-path "$(XML_DATA_DIR)" \
		--target-output-filename-pattern "{document.name}.xml" \
		--limit "$(TRAIN_LIMIT)" \
		--threads 10
	$(RUN_TOOLS) ls -l "$(PDF_DATA_DIR)"
	$(RUN_TOOLS) ls -l "$(XML_DATA_DIR)"


get-example-data-cloud: build
	$(RUN_TOOLS) python -m sciencebeam_trainer_grobid_tools.download_source_files \
		--document-file-list "$(SAMPLE_DATA_PATH)/$(SAMPLE_FILE_LIST)" \
		--document-file-column "source_url" \
		--target-file-list "$(SAMPLE_DATA_PATH)/$(SAMPLE_FILE_LIST)" \
		--target-file-column "xml_url" \
		--document-output-path "$(SAMPLE_CLOUD_DATASET_PATH)/pdf" \
		--document-output-filename-pattern "{name}.pdf.gz" \
		--target-output-path "$(SAMPLE_CLOUD_DATASET_PATH)/xml" \
		--target-output-filename-pattern "{document.name}.xml.gz" \
		--limit "$(TRAIN_LIMIT)" \
		--threads 10
	gsutil ls -l "$(SAMPLE_CLOUD_DATASET_PATH)/pdf/"
	gsutil ls -l "$(SAMPLE_CLOUD_DATASET_PATH)/xml/"


auto-annotate-header: build
	$(RUN_TOOLS) bash -c 'grep "titlePart" "$(DATASET_DIR)/header/corpus/tei-raw/"*.xml'
	$(RUN_TOOLS) python -m sciencebeam_trainer_grobid_tools.auto_annotate_header \
			--source-base-path "$(DATASET_DIR)/header/corpus/tei-raw" \
			--output-path "$(DATASET_DIR)/header/corpus/tei-auto" \
			--xml-path "$(XML_DATA_DIR)" \
			--xml-filename-regex "$(HEADER_XML_TO_TARGET_XML_FILENAME_PATTERN)" \
			--fields title
	$(RUN_TOOLS) bash -c 'grep "titlePart" "$(DATASET_DIR)/header/corpus/tei-auto/"*.xml'


auto-annotate-header-cloud: build
	$(RUN_TOOLS) python -m sciencebeam_trainer_grobid_tools.auto_annotate_header \
			--source-base-path "$(SAMPLE_CLOUD_DATASET_PATH)/header/corpus/tei-raw" \
			--output-path "$(SAMPLE_CLOUD_DATASET_PATH)/header/corpus/tei-auto" \
			--xml-path "$(SAMPLE_CLOUD_DATASET_PATH)/xml" \
			--xml-filename-regex "$(HEADER_XML_TO_TARGET_XML_FILENAME_PATTERN)" \
			--fields title


copy-raw-header-training-data-to-tei: build
	$(RUN_TOOLS) bash -c '\
		mkdir -p "$(DATASET_DIR)/header/corpus/tei" && \
		cp "$(DATASET_DIR)/header/corpus/tei-auto/"*.xml "$(DATASET_DIR)/header/corpus/tei/" \
		'


copy-auto-annotate-header-training-data-to-tei: build
	$(RUN_TOOLS) bash -c '\
		mkdir -p "$(DATASET_DIR)/header/corpus/tei" && \
		cp "$(DATASET_DIR)/header/corpus/tei-auto/"*.xml "$(DATASET_DIR)/header/corpus/tei/" \
		'


copy-auto-annotate-header-training-data-to-tei-cloud: build
	$(RUN) gsutil -m cp \
		"$(SAMPLE_CLOUD_DATASET_PATH)/header/corpus/tei-auto/*.xml.gz" \
		"$(SAMPLE_CLOUD_DATASET_PATH)/header/corpus/tei/"


tools-shell: build
	$(RUN_TOOLS) bash


tools-delete-pyc: build-dev
	$(RUN_TOOLS_DEV) find . -name '*.pyc' -delete


tools-test: build-dev tools-delete-pyc
	$(RUN_TOOLS_DEV) ./project-tests.sh


tools-watch: build-dev tools-delete-pyc
	$(RUN_TOOLS_DEV) pytest-watch


tools-dev-shell: build-dev
	$(RUN_TOOLS_DEV) bash


.grobid-trained-model-image-tag:
	$(eval GROBID_TRAINED_MODEL_IMAGE_TAG = $(GROBID_TAG)-$(SAMPLE_DATA_NAME))
	@echo GROBID_TRAINED_MODEL_IMAGE_TAG=$(GROBID_TRAINED_MODEL_IMAGE_TAG)


grobid-trained-model-build: .grobid-trained-model-image-tag build-docker
	$(DOCKER_COMPOSE) run --rm tools-docker build-grobid-docker-image.sh \
		"$(SAMPLE_CLOUD_MODELS_PATH)" \
		"$(GROBID_REPO):$(GROBID_TAG)" \
		"local/$(GROBID_TRAINED_MODEL_IMAGE_TAG)"


grobid-trained-model-build-cloud: .grobid-trained-model-image-tag build
	$(RUN_TOOLS) build-grobid-docker-image.sh \
		"$(SAMPLE_CLOUD_MODELS_PATH)" \
		"$(GROBID_REPO):$(GROBID_TAG)" \
		"$(GROBID_TRAINED_MODEL_IMAGE_REPO):$(GROBID_TRAINED_MODEL_IMAGE_TAG)" \
		"$(GCP_PROJECT)"


ci-build-and-test:
	make DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" build tools-test


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
