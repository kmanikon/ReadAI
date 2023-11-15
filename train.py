from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
# Additional imports for data preprocessing and training
import pandas as pd
import torch

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")


# starting model (adjust accordingly)
model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base", num_labels=3)  # 3 classes for readability

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)


df_train = pd.read_csv('train.csv')
df_train.head()

# Define readability categories based on target values
def categorize_readability(target):
    if -3.7 <= target <= -2.0:
        return "Elementary"
    elif -2.0 < target <= 0.5:
        return "High School"
    else:
        return "College"

# Apply the categorization to the dataset
df_train['readability_category'] = df_train['target'].apply(categorize_readability)
df_train.head()

label_mapping = {
    "Elementary": 0,
    "High School": 1,
    "College": 2
}

train_labels = ["Elementary", "High School", "College"]

# check if tokenizer works
text = df_train['excerpt'][0]
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Tokenize all
encoded_texts = [tokenizer(text, padding=True, truncation=True, return_tensors="pt") for text in df_train['excerpt']]

#train_labels = torch.tensor([label_mapping[label] for label in train_labels])  # Convert labels to numerical values
train_labels = df_train['readability_category']
train_labels = torch.tensor([label_mapping[label] for label in train_labels])  # Convert labels to numerical values





# Define the loss function (CrossEntropyLoss for classification)
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer (e.g., Adam)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 4  # Adjust as needed
model.train()
for epoch in range(num_epochs):
    total_loss = 0.0

    # progress 0:600
    for i in range(len(encoded_texts)):
        input_ids = encoded_texts[i]['input_ids']
        attention_mask = encoded_texts[i]['attention_mask']
        label = train_labels[i]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label.unsqueeze(0))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print('epoch: ' + str(epoch + 1) + ' iteration: ' + str(i) + '/' + str(len(encoded_texts)))

    average_loss = total_loss / len(encoded_texts)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

# Save the trained model (adjust name)
model.save_pretrained("fine_tuned_bigbird_model3")