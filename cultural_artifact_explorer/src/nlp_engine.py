import yaml
from cultural_artifact_explorer.src.nlp.summarization import TextSummarizer

# Optional imports -- these depend on heavy frameworks like torch which may not
# be available in the testing environment.  Import failures are tolerated and
# the related features will simply be disabled.
try:
    from cultural_artifact_explorer.src.nlp.translation import TextTranslator
except Exception:  # pragma: no cover - translation requires optional deps
    TextTranslator = None

try:
    from cultural_artifact_explorer.src.nlp.ner import NERTagger
except Exception:  # pragma: no cover - NER requires optional deps
    NERTagger = None

class NLPEngine:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            nlp_config = yaml.safe_load(f)
        self.summarizer = TextSummarizer(config_path)
        self.translators = {}
        if TextTranslator is not None:
            for model_key in nlp_config.get('translation', {}).get('models', {}):
                self.translators[model_key] = TextTranslator(config_path, model_key)
        else:  # pragma: no cover - optional feature
            if nlp_config.get('translation'):
                print("Translation support disabled: optional dependencies missing")

        self.ner_tagger = None
        if NERTagger is not None:
            self.ner_tagger = NERTagger(config_path)
        elif nlp_config.get('ner'):
            print("NER support disabled: optional dependencies missing")

    def get_summary(self, text, min_length=None, max_length=None, **kwargs):
        return self.summarizer.summarize(text, min_length, max_length, **kwargs)

    def get_translation(self, text, model_key='en_hi'):
        if not self.translators:
            raise RuntimeError("Translation functionality is not available")
        if model_key not in self.translators:
            raise ValueError(f"Translator for model_key '{model_key}' not found.")
        return self.translators[model_key].translate(text)

    def get_ner(self, text):
        if self.ner_tagger is None:
            raise RuntimeError("NER functionality is not available")
        return self.ner_tagger.extract_entities(text)

def get_nlp_engine(config_path='cultural_artifact_explorer/configs/nlp.yaml'):
    return NLPEngine(config_path)
