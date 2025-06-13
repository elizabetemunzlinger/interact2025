# Eliciting User-Defined Mid-Air Hand Gestures for Hybrid Meeting Platform Control: Results, Insights, and Design Implications

This repository contains the code, data, and analysis for a user study on mid-air hand gestures for controlling hybrid meeting platforms. The project investigates how users define gestures for common meeting actions, analyzes gesture semantics, and provides insights for gesture-based interface design.

## Repository Structure

- `data/` — Processed datasets, including:
  - `d0_hand_gesture_descriptions.csv`: Hand gesture descriptions and their DeBERTa-based embeddings.
  - `d1_d6_*.csv`: Taxonomy classifications (nature, locale) by various LLMs.
- `dataset/` — Raw and reference datasets:
  - `elicit_cam.csv`: Original elicitation data (participant gestures for commands).
  - `opposite_gestures.csv`: Mappings of gestures to their opposites.
- `outputs/` — Analysis results:
  - `elicitation_dataset.csv`: Processed dataset with taxonomy and consensus.
  - `gesture_counts.csv`: Frequency of gestures per command.
- `pgf/` — Plots and visualizations (PGF format for LaTeX).
- `src/` — Python scripts for embedding creation and classification:
  - `s0_create_gesture_embeddings.py`: Generates gesture embeddings using DeBERTa.
  - `s1_s6_*.py`: Scripts for LLM-based taxonomy classification.
- `data_analysis.ipynb` — Main Jupyter notebook for statistical analysis and visualization.
- `pyproject.toml`, `environment.yml` — Dependency management files.

## Data and Embeddings

- **Gesture Descriptions:** Each gesture is described textually and associated with a command (e.g., "Raise Hand", "Mute/Unmute").
- **Embeddings:** Descriptions are embedded using a pre-trained DeBERTa model (Hugging Face Transformers) for semantic analysis and clustering.
- **Taxonomy:** Gestures are classified by nature (iconic, metaphoric, deictic, symbolic, pantomimic) and locale (mouth, ear, eyes, other self, no self interaction) using LLMs (OpenAI, Gemini, Llama).

## Analysis Workflow

1. **Data Loading:** All datasets are loaded and preprocessed in `data_analysis.ipynb`.
2. **Gesture Frequency:** Computes and visualizes the most common gestures per command.
3. **Agreement & Oppositeness:** Calculates agreement rates and analyzes use of opposite gestures.
4. **Consensus & Dissimilarity:** Uses Dynamic Time Warping (DTW) and LDA to quantify and visualize consensus in gesture semantics.
5. **Taxonomy Classification:** Aggregates LLM-based taxonomy labels and computes inter-rater agreement (Fleiss' Kappa).
6. **Visualization:** Generates plots for gesture distributions, taxonomy proportions, and embedding clusters (see `pgf/`).

## Requirements & Installation

- Python >= 3.13

Install dependencies with Poetry or Conda:

```bash
conda env create --file=environment.yml
conda activate interact2025
poetry install
```

## Usage

- To generate gesture embeddings: run `src/s0_create_gesture_embeddings.py`.
- To classify gestures by taxonomy: run scripts in `src/` (e.g., `s1_openai_nature_classifier.py`).
- For full analysis and visualization: open and run `data_analysis.ipynb`.

## Outputs

- Processed datasets and analysis results in `outputs/`.
- Visualizations in `pgf/` (for inclusion in LaTeX documents).

## Citation

If you use this code or data, please cite the associated paper:

> Munzlinger, Elizabete; Narcizo, Fabricio Batista; Briet, Renata; Shimanuki, Mario Tadashi; Vucurevich, Ted; and Hansen, Dan Witzner. (2025). Eliciting User-Defined Mid-Air Hand Gestures for Hybrid Meeting Platform Control: Results, Insights, and Design Implications. In: Human-Computer Interaction – INTERACT 2025. INTERACT 2025. Lecture Notes in Computer Science, vol 12345. Springer, Cham. https://doi.org/00.0000/000-0-000-00000-0_0

## License

MIT License. See `LICENSE` for details.

## Contact

For questions or collaborations, please refer to the original paper or contact the authors via the repository email.

---

**Authors:** Elizabete Munzlinger, Fabricio Batista Narcizo, Renata Briet, Mario Tadashi Shimanuki, Ted Vucurevich, and Dan Witzner Hansen.
