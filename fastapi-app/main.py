from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from PIL import Image
import io
import torch

app = FastAPI()

# Load ResNet pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

@app.get("/")
def read_root():
    return {"message": "ResNet Image Classification API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get the predicted class index
    predicted_idx = output.argmax().item()

    return {"predicted_class_index": predicted_idx}
