ARG base_image
FROM ${base_image}

COPY models /tmp/models

RUN for model_dir in /tmp/models/*; do \
  echo "model dir: $model_dir" && \
  ls -l $model_dir/ && \
  { if [ -f "$model_dir/model.wapiti.gz" ]; then gunzip $model_dir/model.wapiti.gz; fi } && \
  cp $model_dir/model.wapiti ./grobid-home/models/$(basename $model_dir)/ && \
  ls -l ./grobid-home/models/$(basename $model_dir)/; \
  done
