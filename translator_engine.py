from transformers import MarianMTModel, MarianTokenizer
import torch

class MangaTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-ja-en"):
        print(f"Loading translation model: {model_name}...")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def translate(self, text):
        if not text.strip():
            return ""
        
        # Prepare inputs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)
        
        # Decode results
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def translate_batch(self, texts):
        if not texts:
            return []
            
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)
            
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

if __name__ == "__main__":
    # test
    translator = MangaTranslator()
    print(translator.translate("こんにちは、世界！"))
