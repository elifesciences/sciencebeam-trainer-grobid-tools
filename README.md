# ScienceBeam Trainer Tools for GROBID

Whereas [sciencebeam-trainer-grobid](https://github.com/elifesciences/sciencebeam-trainer-grobid) is a lightweight wrapper around [GROBID](https://github.com/kermitt2/grobid), intended to be used as Docker container. This project provides additional tools that can be used to prepare the data for GROBID and complete the process after training (e.g. build a new Docker container with the trained model).

The intention is to use cloud storage as the storage between the steps. But one could also just use a data volume.

## Prerequisites

* [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)

## Recommended

* [Google Gloud SDK](https://cloud.google.com/sdk/docs/) for [gcloud](https://cloud.google.com/sdk/gcloud/)

## Development

### Example End-to-End

```bash
make example-data-processing-end-to-end
```

Uses a sample dataset and trains a GROBID model with it.

Note: the sample dataset is currently not public (but the intention is to provide a public dataset in the future)

### Get Example Data

```bash
make get-example-data
```

Downloads and prepares a sample dataset to the `data` Docker volume.

Note: see above regarding dataset not being public at the moment.

### Generate GROBID Training Data

```bash
make generate-grobid-training-data
```

Converts the previously downloaded PDF from the Data volume to GROBID training data. The `tei` files will be stored in `tei-raw` in the dataset. Training on the raw XML wouldn't be of much use as that the annotations the model already knows. Usually one would review and correct those generated XML files using the [annotation guidelines](https://grobid.readthedocs.io/en/latest/training/General-principles/). The final `tei` files should be stored in the `tei` sub directory of the corpus in the dataset. In our case we will be using auto-annotation using JATS XML.

### Upload Dataset (optional)

```bash
make upload-dataset
```

Uploads the local dataset to the cloud. This allows separating the individual steps.

### Auto-annotate Header

```bash
make auto-annotate-header
```

Auto-annotates the `tei-raw` (produced by the `generate-grobid-training-data`) in combination with the JATS XML. The result is stored in `tei-auto`.

### Copy Raw Header Training Data to TEI

```bash
make copy-auto-annotate-header-training-data-to-tei
```

This copies the generated raw tei XML files in `tei-auto` to `tei`. Alternatively you could review the generated `tei-auto` before copying them over.

### Train Header Model with Dataset

```bash
make train-header-model
```

Trains the model over the dataset produced using the previous steps. The output will be the trained GROBID Header Model.

### Upload Header Model

```bash
make CLOUD_MODELS_PATH=gs://bucket/path/to/model upload-header-model
```

Upload the final header model to a location in the cloud. This is assuming that the credentials are mounted to the container. Because the [Google Gloud SDK](https://cloud.google.com/sdk/docs/) also has some support for AWS' S3, you could also specify an S3 location.
