
import io, base64, numpy as np
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn, os

app = FastAPI(title="Tongue Diagnosis AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

CLASS_NAMES = ["healthy","oral_cancer",
               "tooth_marked","tooth_unmarked"]

CLASS_INFO = {
    "healthy"       :{"description":"No significant tongue abnormalities detected.",
                      "recommendation":"Tongue appears healthy. Maintain regular health checkups.",
                      "severity":"Normal"},
    "oral_cancer"   :{"description":"Potential oral lesion detected on tongue surface.",
                      "recommendation":"Please consult a medical professional immediately.",
                      "severity":"High — Seek medical attention"},
    "tooth_marked"  :{"description":"Tooth marks detected along tongue edges indicating possible spleen qi deficiency.",
                      "recommendation":"Consider dietary adjustments and consult a TCM practitioner.",
                      "severity":"Moderate"},
    "tooth_unmarked":{"description":"No tooth marks detected. Tongue surface appears within normal range.",
                      "recommendation":"Continue monitoring. No immediate intervention required.",
                      "severity":"Low"}
}

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def load_model():
    m = models.efficientnet_b0(weights=None)
    f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3,inplace=True),
        nn.Linear(f,256), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(256,4))
    m.load_state_dict(torch.load(
        "best_model.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()

model   = load_model()
print(f"Model loaded on {DEVICE}")

class GradCAM:
    def __init__(self,model,layer):
        self.model=model; self.g=self.a=None
        layer.register_forward_hook(
            lambda m,i,o:setattr(self,"a",o.detach()))
        layer.register_full_backward_hook(
            lambda m,gi,go:setattr(self,"g",go[0].detach()))
    def generate(self,t,idx=None):
        self.g=self.a=None; o=self.model(t)
        if idx is None: idx=o.argmax(1).item()
        self.model.zero_grad(); o[0,idx].backward()
        if self.g is None: return np.zeros((7,7)),idx
        w=self.g.mean(dim=[2,3],keepdim=True)
        c=torch.relu((w*self.a).sum(1,keepdim=True))
        c=c.squeeze().cpu().numpy()
        if c.ndim==0: c=np.zeros((7,7))
        c-=c.min()
        if c.max()>0: c/=c.max()
        return c,idx

gradcam = GradCAM(model,model.features[-1][0])

def b64(a):
    i=Image.fromarray((a*255).astype(np.uint8))
    b=io.BytesIO(); i.save(b,"PNG")
    return base64.b64encode(b.getvalue()).decode()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()

@app.get("/health")
def health():
    return {"status":"healthy","model":"EfficientNet-B0",
            "accuracy":"90.18%","classes":CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img  = Image.open(
            io.BytesIO(await file.read())).convert("RGB")
        ir   = img.resize((224,224))
        t    = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            o=model(t); p=torch.softmax(o,1)
            c,pi=torch.max(p,1)
        pred = CLASS_NAMES[pi.item()]
        ap   = {n:round(p[0][i].item()*100,2)
                for i,n in enumerate(CLASS_NAMES)}
        t2   = transform(img).unsqueeze(0).to(DEVICE)
        t2.requires_grad_(True)
        cam,_= gradcam.generate(t2,pi.item())
        cr   = np.array(Image.fromarray(
            np.uint8(cam*255)).resize((224,224)))/255.0
        orig = np.array(ir)/255.0
        heat = plt.cm.jet(cr)[:,:,:3]
        over = np.clip(0.5*orig+0.5*heat,0,1)
        info = CLASS_INFO.get(pred,{})
        return JSONResponse({
            "success"          :True,
            "predicted_class"  :pred,
            "confidence"       :round(c.item()*100,2),
            "all_probabilities":ap,
            "description"      :info.get("description",""),
            "recommendation"   :info.get("recommendation",""),
            "severity"         :info.get("severity",""),
            "original_image"   :b64(orig),
            "gradcam_overlay"  :b64(over)})
    except Exception as e:
        return JSONResponse(status_code=500,
            content={"success":False,"error":str(e)})

if __name__=="__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
