#!/usr/bin/env python3
# scripts/run_pipeline.py
# CLI script to run the full artifact processing pipeline (OCR -> NLP -> Retrieval options).

import argparse
import sys
import os
import json

# Adjust path for imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.insert(0, project_root)

try:
    from src.pipeline.artifact_processor import ArtifactProcessor
except ImportError:
    print("Error: Could not import ArtifactProcessor from src.pipeline.")
    print("Please ensure the script is run from the project root, the package is installed, or PYTHONPATH is set correctly.")
    # Fallback dummy class for placeholder execution
    class ArtifactProcessor: # type: ignore
        def __init__(self, ocr_cfg, nlp_cfg, ret_cfg): print(f"Dummy ArtifactProcessor (configs: {ocr_cfg}, {nlp_cfg}, {ret_cfg})")
        def process_artifact_image(self, image_path, perform_ocr=True, perform_nlp=True, perform_retrieval=False, target_translation_lang='en'):
            return {
                'image_path': image_path, 'steps_performed': ['ocr_dummy', 'nlp_dummy' if perform_nlp else None, 'retrieval_dummy' if perform_retrieval else None],
                'ocr': {'raw_text': f"Dummy OCR for {image_path}"},
                'nlp': {'summary': "Dummy summary"} if perform_nlp else {},
                'image_to_text_retrieval': [{'text_info': {'id':'dummy_text'}}] if perform_retrieval else []
            }

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

    print(f"--- Running Full Artifact Processing Pipeline Script (Placeholder) ---")
    print(f"  Image: {args.image}")
    print(f"  OCR Config: {args.config_ocr}")
    print(f"  NLP Config: {args.config_nlp}")
    print(f"  Retrieval Config: {args.config_retrieval}")

    if not os.path.exists(args.image) and not isinstance(ArtifactProcessor, type(lambda:0)): # Check not dummy
        print(f"Error: Input image file not found at {args.image}")
        # For placeholder, allow it to proceed with the path string
        # sys.exit(1)

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
        # try:
        #     # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        #     # with open(args.output_file, 'w', encoding='utf-8') as f_out:
        #     #     f_out.write(output_json_str)
        #     print(f"\nPipeline results saved to: {args.output_file} (placeholder save)")
        # except Exception as e:
        #     print(f"Error saving results to {args.output_file}: {e}")
        #     print("\nPipeline Results (JSON):\n", output_json_str) # Fallback
        print(f"\n(Placeholder) Would save pipeline results to: {args.output_file}")
        print("\nPipeline Results (JSON):\n", output_json_str) # Still print for placeholder
    else:
        print("\nPipeline Results (JSON):\n", output_json_str)

    print("\n--- Full Artifact Processing Pipeline Script Finished ---")

if __name__ == '__main__':
    # Example usage from project root:
    # python scripts/run_pipeline.py --image data/samples/artifact_image.jpg --enable_image_to_text_retrieval --output_file output/pipeline_runs/artifact_analysis.json
    print("Executing scripts.run_pipeline (placeholder script)")
    # Simulate args for direct placeholder run:
    # Ensure dummy files/dirs exist if not using the dummy classes from ImportError block
    # for cfg_dir in ["configs", "output/pipeline_results"]:
    #    if not os.path.exists(cfg_dir): os.makedirs(cfg_dir, exist_ok=True)
    # for cfg_file in ["configs/ocr.yaml", "configs/nlp.yaml", "configs/retrieval.yaml"]:
    #    if not os.path.exists(cfg_file): open(cfg_file, 'a').close()
    # dummy_input_img = "dummy_pipeline_input.png"; open(dummy_input_img, 'a').close()
    # sys.argv = ['', '--image', dummy_input_img, '--enable_image_to_text_retrieval', '--output_file', 'output/pipeline_results/pipeline_test.json']
    # main()
    # if os.path.exists(dummy_input_img): os.remove(dummy_input_img)
    print("To run full placeholder: python scripts/run_pipeline.py --image ./dummy.png --config_ocr configs/ocr.yaml ...")
