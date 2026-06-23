import gradio as gr
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

def ask_question(question: str):
    if not question or question.strip() == "":
        return "Please enter a question.", "", ""
    
    result = rag.query(question, k=5)
    
    
    sources = ""
    for i, meta in enumerate(result["retrieved_metadata"], 1):
        sources += f"**Source {i}** - Complaint ID: {meta.get('complaint_id', 'N/A')} | "
        sources += f"Category: {meta.get('product_category', 'N/A')}\n\n"
    
    return result["answer"], sources, result["retrieved_metadata"]

with gr.Blocks(title="CrediTrust Complaint Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CrediTrust Complaint Intelligence")
    gr.Markdown("Ask questions about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.")
    
    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What are the most common issues with Credit Cards?",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary", size="large")
                clear_btn = gr.Button("Clear", size="large")
        
        with gr.Column(scale=6):
            answer_output = gr.Markdown(label="Answer")
    
    with gr.Accordion(" Retrieved Sources (for verification)", open=False):
        sources_output = gr.Markdown(label="Sources")
  
    submit_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output, gr.State()]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=None,
        outputs=[answer_output, sources_output, question_input]
    )

    gr.Markdown("### Note: Answers are generated based on real customer complaints.")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        
        debug=True
    )