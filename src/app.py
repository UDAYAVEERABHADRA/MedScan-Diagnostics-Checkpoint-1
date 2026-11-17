import gradio as gr
from PIL import Image
from gradcam import generate_gradcam

def predict(image):
    path = "results/gradio_input.png"
    image.save(path)
    heatmap, predicted, probs = generate_gradcam(path)
    labels = ["NORMAL", "PNEUMONIA", "COVID-19"]
    return predicted, {labels[i]: float(probs[i]) for i in range(3)}, Image.fromarray(heatmap)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload chest X-ray"),
    outputs=[
        gr.Textbox(label="Predicted class"),
        gr.Label(label="Class probabilities"),
        gr.Image(label="Grad-CAM heatmap"),
    ],
    title="MedScan Diagnostics — Chest X-Ray Classifier",
    description="Research prototype only. Not for clinical diagnosis."
)

if __name__ == "__main__":
    demo.launch()
