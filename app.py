import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# to generate examples
pastas = ['penne', 'fusilli', 'fettuccine', 'macaroni', 'orecchiette', 'gnocchi']

gr.Interface(
    fn=predict,
    inputs=gr.Image(height=512, width=512),
    outputs=gr.Label(num_top_classes=3),
    title="Pasta Detector",
    description="A pasta classifier trained on duckduckgo searches",
    examples=[f"images/{pasta}" for pasta in pastas]
).launch(share=True)
