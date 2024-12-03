import pandas as pd
from transformers import BertTokenizer
import tensorflow as tf
from transformers import TFBertModel
import time
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import re

import nltk
nltk.download('punkt_tab')

# Load data
initial_train_data = pd.read_csv("cnn_dailymail/train.csv")
initial_val_data = pd.read_csv("cnn_dailymail/validation.csv")
initial_test_data = pd.read_csv("cnn_dailymail/test.csv")

# Use 10% of data
train_data = initial_train_data.sample(frac=0.1, random_state=42)
val_data = initial_val_data.sample(frac=0.1, random_state=42)
# Use 5% of test data
test_data = initial_test_data.sample(frac=0.05, random_state=42)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Split articles into sentences

def tokenize_sentences(article, max_len=512):
    #sentences = re.split(r'(?<!\w\.\w.)(?<!\b[A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s|\\n', article)
    sentences = nltk.sent_tokenize(article)
    sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) >= 3]
    tokenized = [tokenizer(sentence,
                           return_tensors='tf',
                           max_length=max_len,
                           padding='max_length',
                           truncation=True) for sentence in sentences]
    return list(sentences), tokenized

# Preprocess and flatten the data
def preprocess_and_flatten(data, tokenizer):
    flat_data = []
    for _, row in data.iterrows():
        sentences, tokenized = tokenize_sentences(row['article'])
        for i, sentence in enumerate(sentences):
            flat_data.append({
                'article_id': row['id'],
                'sentence': sentence,
                'highlights': row['highlights'],
                'tokenized': tokenized[i]
            })
    return pd.DataFrame(flat_data)

# Start preprocessing
start_time = time.time()
train_flat = preprocess_and_flatten(train_data, tokenizer)
val_flat = preprocess_and_flatten(val_data, tokenizer)
test_flat = preprocess_and_flatten(test_data, tokenizer)
preprocessing_time = time.time() - start_time

print(f"Preprocessing Time: {preprocessing_time:.2f} seconds")

# Save preprocessed data as pickle files
train_flat.to_pickle('preprocessed_train_data.pkl')
val_flat.to_pickle('preprocessed_val_data.pkl')
test_flat.to_pickle('preprocessed_test_data.pkl')

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Label sentences based on ROUGE similarity
def label_sentence(sentence, highlights, threshold_rouge1=0.3):
    scores = scorer.score(sentence, highlights)
    return 1 if scores['rouge1'].fmeasure >= threshold_rouge1 else 0

train_flat['label'] = train_flat.apply(
    lambda row: label_sentence(row['sentence'], row['highlights']), axis=1
)
val_flat['label'] = val_flat.apply(
    lambda row: label_sentence(row['sentence'], row['highlights']), axis=1
)

# Convert data to TensorFlow datasets
def create_tf_dataset(data, batch_size):
    input_ids = tf.concat([item['input_ids'] for item in data['tokenized']], axis=0)
    attention_masks = tf.concat([item['attention_mask'] for item in data['tokenized']], axis=0)
    labels = tf.convert_to_tensor(data['label'].tolist())
    dataset = tf.data.Dataset.from_tensor_slices(
        ({'input_ids': input_ids, 'attention_mask': attention_masks}, labels)
    ).batch(batch_size)
    return dataset

batch_size = 16
train_dataset = create_tf_dataset(train_flat, batch_size)
val_dataset = create_tf_dataset(val_flat, batch_size)

class SentenceRankerModel(tf.keras.Model):
    def __init__(self, bert_model_name='bert-base-uncased'):
        super(SentenceRankerModel, self).__init__()
        self.bert = TFBertModel.from_pretrained(bert_model_name)
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        outputs = self.bert(inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
        scores = self.classifier(cls_output)
        return scores

model = SentenceRankerModel()
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Freeze all layers in BERT
for layer in model.bert.layers:
    layer.trainable = False
for layer in model.bert.layers[-4:]:
    layer.trainable = True

# Train the model
start_time = time.time()
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    verbose=1
)
training_time = time.time() - start_time

print(f"Training Time: {training_time:.2f} seconds")

# Save the model
model.save('trained_model.keras')

# Plot loss and accuracy
def plot_training_history(history, plot_filename='training_history.png'):

    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Loss/Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    # Save the plot as a PNG image
    plt.savefig(plot_filename)
    plt.close()
plot_training_history(history, plot_filename='training_history.png')

def generate_summary(article, model, tokenizer, top_n=3):

    sentences, tokenized = tokenize_sentences(article)
    scores = []
    for t in tokenized:
        inputs = {
            'input_ids': t['input_ids'],
            'attention_mask': t['attention_mask']
        }
        score = model.predict(inputs, verbose=0)
        scores.append(score[0][0])

    top_sentences = [sentences[i] for i in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_n]]
    return " ".join(top_sentences)

def evaluate(dataset, model, tokenizer):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    for _, row in dataset.iterrows():
        generated_summary = generate_summary(row['article'], model, tokenizer)
        score = scorer.score(generated_summary, row['highlights'])
        rouge1_fmeasure = score['rouge1'].fmeasure
        rouge2_fmeasure = score['rouge2'].fmeasure
        rougeL_fmeasure = score['rougeL'].fmeasure

        rouge1_scores.append(rouge1_fmeasure)
        rouge2_scores.append(rouge2_fmeasure)
        rougeL_scores.append(rougeL_fmeasure)
    result_df = pd.DataFrame({
        'rouge1': rouge1_scores,
        'rouge2': rouge2_scores,
        'rougeL': rougeL_scores
    })
    return result_df

test_data.shape

# Evaluate on the  test_data
test_scores = evaluate(test_data, model, tokenizer)

# Plot ROUGE scores
plt.figure(figsize=(10, 8))

# Plot ROUGE-1 scores
plt.plot(range(1, len(test_scores['rouge1']) + 1), test_scores['rouge1'], label='ROUGE-1', linestyle='', marker='o')
# Plot ROUGE-2 scores
plt.plot(range(1, len(test_scores['rouge2']) + 1), test_scores['rouge2'], label='ROUGE-2', linestyle='', marker='o')
# Plot ROUGE-L scores
plt.plot(range(1, len(test_scores['rougeL']) + 1), test_scores['rougeL'], label='ROUGE-L', linestyle='', marker='o')

# Add labels and title
plt.xlabel('Sample Index')
plt.ylabel('ROUGE Score')
plt.title('ROUGE Scores for Extractive Summarization')
plt.legend()

# Save the plot as a PNG image
plot_filename = 'rouge_scores_plot_points.png'
plt.tight_layout()
plt.savefig(plot_filename)

print(f"Preprocessing Time: {preprocessing_time:.2f} seconds")
print(f"Training Time: {training_time:.2f} seconds")
print("Average ROUGE Scores:")
print(f"ROUGE-1: {test_scores['rouge1'].mean():.4f}")
print(f"ROUGE-2: {test_scores['rouge2'].mean():.4f}")
print(f"ROUGE-L: {test_scores['rougeL'].mean():.4f}")

# Extractive Summary Example

example_article = test_data.iloc[0]['article']
print("Article: ")
print(example_article)
example_summary = generate_summary(example_article, model, tokenizer)
print(f"Extractive Summary Example:\n{example_summary}")