# ch-vits-tts
Inference server for Swiss German TTS based on the VITS model, compatible with Amazon SageMaker on AWS.

## Model folder structure
```
📦model
 ┣ 📂t5
 ┃ ┣ 📜config.json
 ┃ ┣ 📜pytorch_model.bin
 ┃ ┣ 📜spiece.model
 ┃ ┣ 📜tokenizer_config.json
 ┃ ┗ 📜tokenizer.json
 ┣ 📂vits
 ┃ ┣ 📜config.yaml
 ┃ ┗ 📜model.pth
 ┗ 📂xvector
   ┗ 📜spk_xvector.ark
```

## Local instance
```
docker build . -t tts
```

```
docker run --gpus=all -p 127.0.0.1:8080:8080 -v <PATH_TO_MODEL_FILES>\model:/opt/ml/model tts
```