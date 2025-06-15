import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants
img_height, img_width = 64, 64
hidden_dim = 64
num_classes = 3
class_names = ["Corrected", "Normal", "Reversal"]

# Model Definition
class LiquidNeuron(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LiquidNeuron, self).__init__()
        self.input_weight = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.hidden_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.time_constant = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.orthogonal_(self.hidden_weight)
        self.time_constant.data.fill_(1.0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_weight.size(0), device=x.device)
        for t in range(seq_len):
            u = torch.matmul(x[:, t, :], self.input_weight)
            h = h + (torch.tanh(u + torch.matmul(h, self.hidden_weight)) - h) / self.time_constant
        return h

class LiquidNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNNModel, self).__init__()
        self.liquid = LiquidNeuron(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), img_height, img_width * 3)
        h = self.liquid(x)
        return self.fc(h)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiquidNNModel(input_dim=img_width*3, hidden_dim=hidden_dim, output_dim=num_classes)
model_path = os.path.join('model',r'C:\Users\Prasad\Desktop\dy_1\model\dyslexia_lnn_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred = torch.argmax(probabilities, dim=1).item()
            prediction = class_names[pred]
            confidence = float(probabilities[0][pred]) * 100

        dyslexic_classes = ["Reversal", "Corrected"]
        is_dyslexic = prediction in dyslexic_classes

        return render_template(
            'result.html',
            prediction=prediction,
            confidence=f"{confidence:.2f}",
            is_dyslexic=is_dyslexic,
            image_url=os.path.join('uploads', filename)
        )
    except Exception as e:
        print(f"Error processing image: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)