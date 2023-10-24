import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from nltk.tokenize import word_tokenize
import random

# Download pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
# Remove the classification layer
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
# Set ResNet in evaluation mode
resnet.eval()

# Define an RNN-based captioning model
class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(CaptioningModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

# Load a pre-trained captioning model
# You can train your own model on a dataset like COCO
# or use an existing pre-trained model.

# Define functions to preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

# Generate captions for an image
def generate_caption(image_path, model, vocab, max_length=20):
    image = preprocess_image(image_path)
    image_features = resnet(image).squeeze().unsqueeze(0)
    model.eval()
    inputs = torch.LongTensor([vocab['<start>']])
    caption = ['<start>']

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(image_features, inputs)
            _, predicted = outputs.max(2)
            word = predicted.item()
            caption.append(vocab.idx2word[word])
            inputs = predicted

            if word == vocab['<end>']:
                break

    return ' '.join(caption[1:-1])

# Define the vocabulary
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

vocab = Vocabulary()
vocab.add_word('<start>')
vocab.add_word('<end>')

# Load a pre-trained vocabulary (you can train your own vocabulary as well)

# Example usage
image_path = 'C:\Users\SHUBHAM SANGER\Desktop\image_captioning\1.jpg'
caption = generate_caption(image_path, captioning_model, vocab)
print("Generated Caption:", caption)