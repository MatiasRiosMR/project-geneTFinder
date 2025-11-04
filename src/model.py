import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinClassifier(nn.Module):
    """
    Modelo CNN + BiLSTM para clasificación de proteínas
    """
    def __init__(self, vocab_size=21, embedding_dim=128, hidden_dim=256, 
                 num_filters=128, dropout=0.3):
        super(ProteinClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers con diferentes tamaños de kernel
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=7, padding=3)
        
        self.batch_norm = nn.BatchNorm1d(num_filters * 3)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(num_filters * 3, hidden_dim, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=dropout)
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def attention_layer(self, lstm_output):
        """
        Attention mechanism
        """
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # Transpose for conv1d: [batch, embedding_dim, seq_len]
        embedded_t = embedded.transpose(1, 2)
        
        # Parallel convolutions
        conv1_out = F.relu(self.conv1(embedded_t))
        conv2_out = F.relu(self.conv2(embedded_t))
        conv3_out = F.relu(self.conv3(embedded_t))
        
        # Concatenate conv outputs
        conv_out = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        conv_out = self.batch_norm(conv_out)
        
        # Transpose back for LSTM: [batch, seq_len, features]
        conv_out = conv_out.transpose(1, 2)
        conv_out = self.dropout(conv_out)
        
        # BiLSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # Attention pooling
        attended = self.attention_layer(lstm_out)
        attended = self.dropout(attended)
        
        # Fully connected layers
        fc1_out = F.relu(self.fc1(attended))
        fc1_out = self.dropout(fc1_out)
        
        fc2_out = F.relu(self.fc2(fc1_out))
        fc2_out = self.dropout(fc2_out)
        
        output = torch.sigmoid(self.fc3(fc2_out))
        
        return output.squeeze()

class SimpleProteinClassifier(nn.Module):
    """
    Modelo más simple para pruebas rápidas
    """
    def __init__(self, vocab_size=21, embedding_dim=64, hidden_dim=128, dropout=0.3):
        super(SimpleProteinClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        
        fc1_out = F.relu(self.fc1(hidden))
        fc1_out = self.dropout(fc1_out)
        
        output = torch.sigmoid(self.fc2(fc1_out))
        return output.squeeze()
