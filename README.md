# Sentiment Analysis with Pre-trained Transformer Models

## Objective
Build a sentiment analysis model using pre-trained transformer models.

## Steps
1. Data Collection
2. Model Implementation
3. Evaluation
4. Deployment

## How to Run

### Training and Evaluation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Place your dataset CSV file in the `data` folder. Ensure the file is named `emotions.csv` and has at least two columns: `text` and `label`.
3. Run the training and evaluation script:
    ```bash
    python scripts/train_and_evaluate.py
    ```

### Deployment
1. Update the `model` path in `scripts/app.py` to the actual path where your trained model is saved.
2. Run the Streamlit app:
    ```bash
    streamlit run scripts/app.py
    ```

## Results
Include evaluation metrics and example outputs here.

