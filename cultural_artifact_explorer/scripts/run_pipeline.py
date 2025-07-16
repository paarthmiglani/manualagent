#!/usr/bin/env python3
# scripts/run_pipeline.py
# CLI script to run the full artifact processing pipeline (OCR -> NLP -> Retrieval options).

import argparse
import sys
import os
import json

# Adjust path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.pipeline.artifact_processor import ArtifactProcessor
except ImportError:
    print("Error: Could not import ArtifactProcessor from src.pipeline.")
    print("Please ensure the script is run from the project root, the package is installed, or PYTHONPATH is set correctly.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full artifact processing pipeline on an image.")
    parser.add_argument('--image', type=str, required=True,
                        help="Path to the artifact image file to process.")
    parser.add_argument('--config_ocr', type=str, default="configs/ocr.yaml",
                        help="Path to OCR configuration.")
    parser.add_argument('--config_nlp', type=str, default="configs/nlp.yaml",
                        help="Path to NLP configuration.")
    parser.add_argument('--config_retrieval', type=str, default="configs/retrieval.yaml",
                        help="Path to Retrieval configuration.")
    parser.add_argument('--output_file', type=str, default=None,
                        help="File to save the JSON results. Prints to console if not specified.")

    # Pipeline step control
    parser.add_argument('--skip_ocr', action='store_true', help="Skip the OCR step.")
    parser.add_argument('--skip_nlp', action='store_true', help="Skip all NLP steps (translation, summary, NER).")
    parser.add_argument('--enable_image_to_text_retrieval', action='store_true',
                        help="Enable image-to-text retrieval (find related texts for the input image).")
    parser.add_argument('--translate_to', type=str, default='en',
                        help="Target language for translation if NLP is performed.")

    args = parser.parse_args()

    print(f"--- Running Full Artifact Processing Pipeline Script ---")
    print(f"  Image: {args.image}")
    print(f"  OCR Config: {args.config_ocr}")
    print(f"  NLP Config: {args.config_nlp}")
    print(f"  Retrieval Config: {args.config_retrieval}")

    if not os.path.exists(args.image):
        print(f"Error: Input image file not found at {args.image}")
        sys.exit(1)

    try:
        processor = ArtifactProcessor(
            ocr_config_path=args.config_ocr,
            nlp_config_path=args.config_nlp,
            retrieval_config_path=args.config_retrieval
        )
    except Exception as e:
        print(f"Error initializing ArtifactProcessor: {e}")
        sys.exit(1)

    print("\n  Pipeline Settings:")
    print(f"    Perform OCR: {not args.skip_ocr}")
    print(f"    Perform NLP: {not args.skip_nlp}")
    print(f"    Perform Image-to-Text Retrieval: {args.enable_image_to_text_retrieval}")
    if not args.skip_nlp:
        print(f"    Target Translation Language: {args.translate_to}")

    try:
        pipeline_results = processor.process_artifact_image(
            image_path=args.image,
            perform_ocr=(not args.skip_ocr),
            perform_nlp=(not args.skip_nlp),
            perform_retrieval=args.enable_image_to_text_retrieval,
            target_translation_lang=args.translate_to
        )
    except Exception as e:
        print(f"Error during pipeline processing for {args.image}: {e}")
        sys.exit(1)

    # Output results
    output_json_str = json.dumps(pipeline_results, indent=2, ensure_ascii=False)

    if args.output_file:
        try:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, 'w', encoding='utf-8') as f_out:
                f_out.write(output_json_str)
            print(f"\nPipeline results saved to: {args.output_file}")
        except Exception as e:
            print(f"Error saving results to {args.output_file}: {e}")
            print("\nPipeline Results (JSON):\n", output_json_str) # Fallback
    else:
        print("\nPipeline Results (JSON):\n", output_json_str)

    print("\n--- Full Artifact Processing Pipeline Script Finished ---")

if __name__ == '__main__':
    # Example usage from project root:
    # python scripts/run_pipeline.py --image data/samples/artifact_image.jpg --enable_image_to_text_retrieval --output_file output/pipeline_runs/artifact_analysis.json
    main()
