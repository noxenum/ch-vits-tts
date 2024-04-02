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

## Convert T5 model to CT2
```
ct2-transformers-converter --model <PATH_TO_MODEL_FILES>\model\t5 --quantization float16 --output_dir <PATH_TO_MODEL_FILES>\model\t5-ct2
```

## Model folder structure (CT2)
```
📦model
 ┣ 📂t5-ct2
 ┃ ┣ 📜config.json
 ┃ ┣ 📜model.bin
 ┃ ┗ 📜shared_vocabulary.json
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