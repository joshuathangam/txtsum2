import gradio as gr
from transformers import pipeline

# Load the summarization pipeline
pipe = pipeline("summarization", model="facebook/bart-large-cnn")

# Define a function that uses the pipeline
def summarize(text):
    summary = pipe(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Gradio interface
iface = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="Enter text to summarize here..."),
    outputs="text",
    title="Text Summarizer",
    description="Summarize long pieces of text using BART-large-CNN."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
