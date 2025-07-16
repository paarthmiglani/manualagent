import yaml
from cultural_artifact_explorer.src.nlp.summarization import TextSummarizer
from cultural_artifact_explorer.src.nlp.translation import TextTranslator
from cultural_artifact_explorer.src.nlp.ner import NERTagger

class NLPEngine:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            nlp_config = yaml.safe_load(f)
        self.summarizer = TextSummarizer(config_path)
        self.translators = {}
        for model_key in nlp_config.get('translation', {}).get('models', {}):
            self.translators[model_key] = TextTranslator(config_path, model_key)
        self.ner_tagger = NERTagger(config_path)

    def get_summary(self, text, min_length=None, max_length=None, **kwargs):
        return self.summarizer.summarize(text, min_length, max_length, **kwargs)

    def get_translation(self, text, model_key='en_hi'):
        if model_key not in self.translators:
            raise ValueError(f"Translator for model_key '{model_key}' not found.")
        return self.translators[model_key].translate(text)

    def get_ner(self, text):
        return self.ner_tagger.extract_entities(text)

def get_nlp_engine(config_path='cultural_artifact_explorer/configs/nlp.yaml'):
    return NLPEngine(config_path)
