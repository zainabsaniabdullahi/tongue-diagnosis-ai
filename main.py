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
import uvicorn, os, asyncio

app = FastAPI(title="Tongue Diagnosis AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

CLASS_NAMES = [
    "healthy", "oral_cancer",
    "tooth_marked", "tooth_unmarked"
]

CLASS_INFO = {
    "healthy": {
        "description"   : "No significant tongue abnormalities detected.",
        "recommendation": "Tongue appears healthy. Maintain regular health checkups.",
        "severity"      : "Normal"
    },
    "oral_cancer": {
        "description"   : "Potential oral lesion detected on tongue surface.",
        "recommendation": "Please consult a medical professional immediately.",
        "severity"      : "High — Seek medical attention"
    },
    "tooth_marked": {
        "description"   : "Tooth marks detected along tongue edges indicating possible spleen qi deficiency.",
        "recommendation": "Consider dietary adjustments and consult a TCM practitioner.",
        "severity"      : "Moderate"
    },
    "tooth_unmarked": {
        "description"   : "No tooth marks detected. Tongue surface appears within normal range.",
        "recommendation": "Continue monitoring. No immediate intervention required.",
        "severity"      : "Low"
    }
}

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ── Load model once at startup ────────────────────────────────
print("Loading model...")
_model = None

def get_model():
    global _model
    if _model is None:
        m = models.efficientnet_b0(weights=None)
        f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(f, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )
        m.load_state_dict(
            torch.load(
                "best_model.pth",
                map_location=DEVICE,
                weights_only=True
            )
        )
        m.eval()
        _model = m
        print("Model loaded successfully!")
    return _model

# Load at startup so first request is fast
get_model()

# ── Helper: encode image to base64 ───────────────────────────
def encode_image(arr):
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()

@app.get("/health")
def health():
    return {
        "status"  : "healthy",
        "model"   : "EfficientNet-B0",
        "accuracy": "90.18%",
        "device"  : str(DEVICE),
        "classes" : CLASS_NAMES
    }

@app.get("/ping")
def ping():
    return {"message": "pong", "status": "alive"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ── Read image ────────────────────────────────────────
        content = await file.read()
        img     = Image.open(
            io.BytesIO(content)
        ).convert("RGB")
        img_224 = img.resize((224, 224))

        # ── Prediction ────────────────────────────────────────
        m      = get_model()
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = m(tensor)
            probs  = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)

        pred      = CLASS_NAMES[pred_idx.item()]
        confidence= round(conf.item() * 100, 2)
        all_probs = {
            name: round(probs[0][i].item() * 100, 2)
            for i, name in enumerate(CLASS_NAMES)
        }

        # ── Grad-CAM ──────────────────────────────────────────
        acts = []
        grads = []

        def fwd(mod, inp, out):
            acts.append(out.detach())

        def bwd(mod, gin, gout):
            grads.append(gout[0].detach())

        layer = m.features[-1][0]
        h1    = layer.register_forward_hook(fwd)
        h2    = layer.register_full_backward_hook(bwd)

        t2    = transform(img).unsqueeze(0).to(DEVICE)
        t2.requires_grad_(True)
        out2  = m(t2)
        m.zero_grad()
        out2[0, pred_idx.item()].backward()

        h1.remove()
        h2.remove()

        if grads and acts:
            w   = grads[0].mean(dim=[2, 3], keepdim=True)
            cam = torch.relu(
                (w * acts[0]).sum(1, keepdim=True)
            ).squeeze().cpu().numpy()
            if cam.ndim == 0:
                cam = np.zeros((7, 7))
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
        else:
            cam = np.zeros((7, 7))

        # ── Create overlay ────────────────────────────────────
        cam_r = np.array(
            Image.fromarray(
                np.uint8(cam * 255)
            ).resize((224, 224))
        ) / 255.0

        orig    = np.array(img_224) / 255.0
        heatmap = plt.cm.jet(cam_r)[:, :, :3]
        overlay = np.clip(0.5 * orig + 0.5 * heatmap, 0, 1)

        info = CLASS_INFO.get(pred, {})

        return JSONResponse({
            "success"          : True,
            "predicted_class"  : pred,
            "confidence"       : confidence,
            "all_probabilities": all_probs,
            "description"      : info.get("description", ""),
            "recommendation"   : info.get("recommendation", ""),
            "severity"         : info.get("severity", ""),
            "original_image"   : encode_image(orig),
            "gradcam_overlay"  : encode_image(overlay)
        })

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=120
    )
