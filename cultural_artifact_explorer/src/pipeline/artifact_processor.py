# src/pipeline/artifact_processor.py
# Orchestrates the full OCR -> NLP -> Retrieval pipeline for an artifact (image + text)

import yaml
from ocr.ocr_engine import get_ocr_engine
from ..nlp_engine import get_nlp_engine

class ArtifactProcessor:
    def __init__(self, ocr_config_path, nlp_config_path, retrieval_config_path):
        """
        Initializes all components of the artifact processing pipeline.
        Args:
            ocr_config_path (str): Path to OCR configuration.
            nlp_config_path (str): Path to NLP configuration.
            retrieval_config_path (str): Path to Retrieval configuration.
        """
        print("Initializing ArtifactProcessor...")

        self.ocr_config_path = ocr_config_path
        self.nlp_config_path = nlp_config_path
        self.retrieval_config_path = retrieval_config_path
        print("  Configurations paths loaded.")

        self.ocr_engine = get_ocr_engine(self.ocr_config_path)
        self.nlp_engine = get_nlp_engine(self.nlp_config_path)
        self.retriever = "dummy_multimodal_retriever_object" # Placeholder for the search component

        print("  All pipeline components initialized.")

    def process_artifact_image(self, image_path, perform_ocr=True, perform_nlp=True, perform_retrieval=False, target_translation_lang='en'):
        """
        Processes an artifact image through the full pipeline.
        Args:
            image_path (str): Path to the artifact image.
            perform_ocr (bool): Whether to run OCR.
            perform_nlp (bool): Whether to run NLP tasks on OCRed text.
            perform_retrieval (bool): Whether to perform image-to-text retrieval.
            target_translation_lang (str): Target language for translation.
        Returns:
            dict: A dictionary containing all processing results.
        """
        print(f"\nProcessing artifact image: {image_path}...")
        results = {'image_path': image_path, 'steps_performed': []}

        # 1. OCR (if enabled)
        ocr_text_aggregated = None
        if perform_ocr:
            print("  Step 1: Performing OCR...")
            ocr_text_aggregated, _ = self.ocr_engine.get_text(image_path)
            results['ocr'] = {
                'raw_text': ocr_text_aggregated,
            }
            results['steps_performed'].append('ocr')
            print(f"    OCR Result: '{ocr_text_aggregated[:100]}...'")

        # 2. NLP (if enabled and OCR text available)
        if perform_nlp and ocr_text_aggregated:
            print("  Step 2: Performing NLP tasks...")
            nlp_results = {}

            translated_text = self.nlp_engine.get_translation(ocr_text_aggregated, model_key=f"en_{target_translation_lang}")
            nlp_results['translation_to_english'] = translated_text
            print(f"    Translation: '{translated_text[:100]}...'")

            summary = self.nlp_engine.get_summary(translated_text)
            nlp_results['summary'] = summary
            print(f"    Summary: '{summary[:100]}...'")

            entities = self.nlp_engine.get_ner(translated_text)
            nlp_results['named_entities'] = entities
            print(f"    NER: Found {len(entities)} entities. Example: {entities[0] if entities else 'None'}")

            results['nlp'] = nlp_results
            results['steps_performed'].append('nlp')

        # 3. Retrieval (Image-to-Text, if enabled)
        if perform_retrieval:
            print("  Step 3: Performing Image-to-Text Retrieval (placeholder)...")
            retrieved_texts_info = [
                {'text_info': {'id': 'txt_001', 'content': 'A description of a similar artifact...'}, 'score': 0.92},
                {'text_info': {'id': 'txt_005', 'content': 'Another related text entry...'}, 'score': 0.88}
            ] # Dummy
            results['image_to_text_retrieval'] = retrieved_texts_info
            results['steps_performed'].append('image_to_text_retrieval')
            print(f"    Retrieved {len(retrieved_texts_info)} texts for the image (dummy).")

        print(f"Artifact image processing finished for: {image_path}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Process an artifact image through the full pipeline.")
    parser.add_argument('--image', type=str, required=True, help="Path to the artifact image file.")
    parser.add_argument('--ocr_config', type=str, default='configs/ocr.yaml', help="Path to OCR config.")
    parser.add_argument('--nlp_config', type=str, default='configs/nlp.yaml', help="Path to NLP config.")
    parser.add_argument('--retrieval_config', type=str, default='configs/retrieval.yaml', help="Path to Retrieval config.")
    parser.add_argument('--no_ocr', action='store_true', help="Skip OCR step.")
    parser.add_argument('--no_nlp', action='store_true', help="Skip NLP step.")
    parser.add_argument('--enable_retrieval', action='store_true', help="Enable image-to-text retrieval.")

    args = parser.parse_args()

    print("--- Initializing Full Artifact Processing Pipeline (Placeholder Execution) ---")
    # Ensure dummy config files exist for placeholder execution if paths are default
    # For real execution, these files must be properly configured.
    # Example: open(args.ocr_config, 'a').close() # Create if not exists

    processor = ArtifactProcessor(
        ocr_config_path=args.ocr_config,
        nlp_config_path=args.nlp_config,
        retrieval_config_path=args.retrieval_config
    )

    print("\n--- Starting Artifact Processing ---")
    processing_results = processor.process_artifact_image(
        image_path=args.image,
        perform_ocr=not args.no_ocr,
        perform_nlp=not args.no_nlp,
        perform_retrieval=args.enable_retrieval
    )

    print("\n--- Processing Results ---")
    # import json
    # print(json.dumps(processing_results, indent=2, ensure_ascii=False)) # Pretty print results
    # Simplified print for placeholder:
    print(f"Results for image: {processing_results.get('image_path')}")
    print(f"Steps performed: {processing_results.get('steps_performed')}")
    if 'ocr' in processing_results:
        print(f"  OCR Text (first 50 chars): {processing_results['ocr']['raw_text'][:50]}...")
    if 'nlp' in processing_results:
        print(f"  NLP Summary (first 50 chars): {processing_results['nlp']['summary'][:50]}...")
        print(f"  NLP Entities (count): {len(processing_results['nlp']['named_entities'])}")
    if 'image_to_text_retrieval' in processing_results:
        print(f"  Retrieved Texts (count): {len(processing_results['image_to_text_retrieval'])}")

    print("--- End of Placeholder Execution ---")

if __name__ == '__main__':
    # To run this placeholder:
    # python src/pipeline/artifact_processor.py --image path/to/your/image.jpg --enable_retrieval
    # (Ensure dummy config files like configs/ocr.yaml, etc. exist or provide paths)
    print("Executing src.pipeline.artifact_processor (placeholder script)")
    # Example of simulating args:
    # import sys, os
    # # Create dummy configs if they don't exist for placeholder run
    # for cfg in ['configs/ocr.yaml', 'configs/nlp.yaml', 'configs/retrieval.yaml']:
    #   os.makedirs(os.path.dirname(cfg), exist_ok=True)
    #   if not os.path.exists(cfg): open(cfg, 'a').close()
    # # Create a dummy image file
    # dummy_image_file = "dummy_artifact.png"; open(dummy_image_file, 'a').close()
    # sys.argv = ['', '--image', dummy_image_file, '--enable_retrieval']
    # main()
    # if os.path.exists(dummy_image_file): os.remove(dummy_image_file)
    print("To run full placeholder main: python src/pipeline/artifact_processor.py --image your_image.jpg --ocr_config path/to/ocr.yaml ...")
