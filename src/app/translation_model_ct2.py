import ctranslate2
import torch
from transformers.models.t5.tokenization_t5 import T5Tokenizer


class TranslationModelCT2:
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tokenizer = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            print("Started translation model loading...")

            model_path = "/opt/ml/model/t5-ct2"

            cls.model = ctranslate2.Translator(model_path, device=cls.device_name)
            cls.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        return cls.model, cls.tokenizer

    @classmethod
    def predict(cls, dialect: str, text_de: str, beam_size: int = 1) -> str:
        model, tokenizer = cls.get_model()

        input_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(f"{dialect}: {text_de}")
        )

        results = model.translate_batch(
            [input_tokens],
            max_input_length=256,
            beam_size=beam_size,
            num_hypotheses=1,
        )

        output_tokens = results[0].hypotheses[0]

        pred = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True
        )
        return pred
