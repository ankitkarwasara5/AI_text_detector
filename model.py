import torch
from transformers import BertModel

# We'll use the same model name variable for consistency
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
N_CLASSES = 2 # We have two output classes: Human-written (0) and AI-generated (1)

class BERTClassifier(torch.nn.Module):
    """
    The classifier model which wraps the pre-trained BERT model.
    """
    def __init__(self, n_classes):
        """
        Initializes the model layers.
        n_classes (int): The number of output classes.
        """
        super(BERTClassifier, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        # Add a dropout layer to prevent overfitting
        self.drop = torch.nn.Dropout(p=0.3)
        
        # Add a final linear layer for classification
        # BERT-base's hidden size is 768
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass of the model.
        
        Args:
            input_ids: The input token IDs.
            attention_mask: The attention mask to ignore padding tokens.
            
        Returns:
            The output from the final linear layer (raw logits).
        """
        # Pass inputs through the BERT model
        # We are interested in the pooled_output for classification
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Set to False to get tuple output
        )
        
        # Apply dropout to the pooled output
        output = self.drop(pooled_output)
        
        # Pass the result through our final classification layer
        return self.out(output)

# --- How to use it ---
if __name__ == '__main__':
    # Create an instance of the model
    model = BERTClassifier(n_classes=N_CLASSES)
    
    # You can print the model architecture to verify it
    print("--- Model Architecture ---")
    print(model)
    
    # You can also test with some dummy data
    # Create a dummy batch (batch_size=2, max_len=12)
    dummy_input_ids = torch.randint(0, 30000, (2, 12)) # 2 sentences, 12 tokens each
    dummy_attention_mask = torch.ones((2, 12))
    
    # Get model output
    outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
    
    print("\n--- Dummy Output ---")
    print("Output shape:", outputs.shape) # Should be [batch_size, n_classes] -> [2, 2]
    print("Example output:", outputs)