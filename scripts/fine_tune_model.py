from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# Load a pretrained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased')

# Load your custom dataset
dataset = load_dataset('your_custom_dataset')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

# Fine-tune the model
trainer.train()
