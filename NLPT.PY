# Text Summarization Tool using NLP

from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization")

# Sample input text (you can replace this with any long article)
input_text = """
Artificial Intelligence (AI) is transforming the world at an incredible pace. From healthcare to finance,
transportation to education, AI technologies are being adopted to improve efficiency, accuracy, and outcomes.
AI encompasses machine learning, deep learning, natural language processing, and more. Companies are investing heavily
in AI research to stay competitive and innovative. While AI brings many benefits, it also raises ethical concerns
such as bias, data privacy, and job displacement. Governments and organizations are working to establish
regulations and frameworks to ensure responsible AI development. As AI continues to evolve, its integration
into our daily lives will only deepen, making it one of the most important technological trends of the 21st century.
"""

# Generate the summary
summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)

# Display the result
print("Original Text:\n", input_text)
print("\n--- Summary ---\n")
print(summary[0]['summary_text'])
