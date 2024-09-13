import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def main():
    # Load dataset from CSV
    df = pd.read_csv('data/emotions.csv')

    # Check the columns of the DataFrame
    print("Columns in the dataset:", df.columns)

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=2)  # Reduce number of processes

    # Set format for PyTorch
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Split dataset into train and test
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Load pre-trained model with the correct number of labels
    num_labels = 6  # Since we have six categories
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=8,   # Increase batch size
        per_device_eval_batch_size=8,    # Increase batch size
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        dataloader_num_workers=2         # Reduce number of data loading workers
    )

    # Define compute metrics function
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            'accuracy': accuracy_score(p.label_ids, preds),
            'f1': f1_score(p.label_ids, preds, average='weighted')
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Move model to the appropriate device
    model.to(device)

    # Train the model
    trainer.train()

    # Save the final model and tokenizer
    final_model_path = './results/final_model'
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(eval_result)

if __name__ == '__main__':
    main()