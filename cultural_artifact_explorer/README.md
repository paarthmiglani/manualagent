# Cultural Artifact Explorer – Multimodal Retrieval System

This project builds a custom system for exploring Indian cultural artifacts using manually trained models for OCR, NLP, and multimodal retrieval. The goal is to understand images and associated texts of cultural heritage objects through an integrated pipeline that processes visual and textual data.

> ✅ No off-the-shelf models are used — all components are trained from scratch or fine-tuned on curated datasets.

## Project Structure

(A more detailed structure will be maintained here, reflecting the actual layout under `src/`, `data/`, etc.)

## Core Modules
- **OCR (`src/ocr/`)**: Custom-trained Indic OCR.
- **NLP (`src/nlp/`)**: Translation, summarization, NER.
- **Retrieval (`src/retrieval/`)**: Image-text embedding and search.
- **Pipeline (`src/pipeline/`)**: Orchestrates the full workflow.
- **Interface (`src/interface/`)**: User interface (e.g., Streamlit).

## Setup & Running

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cultural_artifact_explorer
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # If using setup.py for editable install:
    # pip install -e .
    ```

4.  **Download/Prepare Data & Models:**
    (Instructions for acquiring necessary datasets and pre-trained model files will be added here. This might involve running scripts from `scripts/` or manually placing files in `data/` and `models/`.)

5.  **Run the application or scripts:**
    -   Launch the Streamlit app: `sh scripts/launch_app.sh` or `streamlit run src/interface/streamlit_app.py`
    -   Use individual pipeline scripts: `python scripts/run_ocr.py --input <image_path>`

## Development

-   **Configuration**: Modify YAML files in `configs/` for training and inference parameters.
-   **Training**: Use scripts in `src/<module>/train.py` or higher-level scripts in `scripts/`.
-   **Testing**: Run tests using `pytest tests/`.

## TODO

-   Finalize dataset paths and model storage strategy.
-   Implement all placeholder scripts and modules.
-   Add comprehensive unit and integration tests.
-   Refine UI/UX.
