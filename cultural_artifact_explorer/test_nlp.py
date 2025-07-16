import unittest
import yaml
import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src import nlp_engine

class TestNLPEngine(unittest.TestCase):
    def setUp(self):
        self.config_path = 'test_nlp_config.yaml'
        self.model_dir = 'test_nlp_models'
        os.makedirs(self.model_dir, exist_ok=True)

        # Create dummy models and configs
        self.create_dummy_files()

        self.nlp_engine = nlp_engine.NLPEngine(self.config_path)

    def create_dummy_files(self):
        # Create dummy summarizer model
        summ_model_path = os.path.join(self.model_dir, 'summarizer.pth')
        with open(summ_model_path, 'w') as f:
            f.write('dummy summarizer model')

        # Create dummy translator model
        trans_model_path = os.path.join(self.model_dir, 'translator.pth')
        with open(trans_model_path, 'w') as f:
            f.write('dummy translator model')

        # Create dummy ner model
        ner_model_path = os.path.join(self.model_dir, 'ner.pth')
        with open(ner_model_path, 'w') as f:
            f.write('dummy ner model')

        # Create dummy vocab
        src_vocab_path = os.path.join(self.model_dir, 'src_vocab.json')
        tgt_vocab_path = os.path.join(self.model_dir, 'tgt_vocab.json')
        word_to_ix_path = os.path.join(self.model_dir, 'word_to_ix.json')
        tag_to_ix_path = os.path.join(self.model_dir, 'tag_to_ix.json')

        with open(src_vocab_path, 'w') as f:
            json.dump({'<PAD>': 0, '<UNK>': 1, 'hello': 2, 'world': 3}, f)
        with open(tgt_vocab_path, 'w') as f:
            json.dump({'<PAD>': 0, '<UNK>': 1, 'hola': 2, 'mundo': 3}, f)
        with open(word_to_ix_path, 'w') as f:
            json.dump({'<PAD>': 0, '<UNK>': 1, 'Taj': 2, 'Mahal': 3, 'is': 4, 'in': 5, 'Agra': 6}, f)
        with open(tag_to_ix_path, 'w') as f:
            json.dump({'O': 0, 'B-LOC': 1, 'I-LOC': 2}, f)

        config = {
            'summarization': {
                'model_path': summ_model_path,
                'tokenizer_path': summ_model_path,
            },
            'translation': {
                'models': {
                    'en_es': {
                        'model_path': trans_model_path,
                        'src_vocab_path': src_vocab_path,
                        'tgt_vocab_path': tgt_vocab_path,
                        'model_params': {
                            'num_encoder_layers': 1,
                            'num_decoder_layers': 1,
                            'embedding_dim': 1,
                            'num_heads': 1,
                        }
                    }
                }
            },
            'ner': {
                'model_path': ner_model_path,
                'vocab_map_path': word_to_ix_path,
                'tag_map_path': tag_to_ix_path,
                'model_params': {
                    'embedding_dim': 1,
                    'hidden_dim': 1,
                }
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def tearDown(self):
        os.remove(self.config_path)
        for root, _, files in os.walk(self.model_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(self.model_dir)

    def test_get_summary(self):
        summary = self.nlp_engine.get_summary("This is a long text")
        self.assertIsInstance(summary, str)

    @unittest.skip("Skipping translation test because it requires torch")
    def test_get_translation(self):
        translation = self.nlp_engine.get_translation("hello world", model_key='en_es')
        self.assertIsInstance(translation, str)

    @unittest.skip("Skipping NER test because it requires torch")
    def test_get_ner(self):
        entities = self.nlp_engine.get_ner("Taj Mahal is in Agra")
        self.assertIsInstance(entities, list)

if __name__ == '__main__':
    unittest.main()
