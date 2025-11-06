# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 06-11-2025
# Register no: 212224040140
# Aim: 
  Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

# AI Tools Required:
   - OpenAI API (GPT Models)
   - Hugging Face Transformers Library


# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that.

In this experiment, we simulate a programmer persona who wants to evaluate how different AI models perform on the same task — summarizing a given article.
The Python program:

- Connects to multiple AI tools (OpenAI and Hugging Face).

- Sends the same text input to both.

- Compares their outputs automatically.

- Generates insights based on length, coherence, and similarity.

This approach helps in AI benchmarking, multi-model decision making, and toolchain automation — skills highly relevant in modern AI development.

# python code :
```
from openai import OpenAI
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

openai_client = OpenAI(api_key="your_openai_api_key_here")


hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


input_text = """
Artificial Intelligence (AI) is transforming industries across the world. 
From healthcare to transportation, AI applications are enhancing efficiency, 
accuracy, and innovation. However, ethical considerations and responsible 
use remain crucial for sustainable AI adoption.
"""

gpt_response = openai_client.responses.create(
    model="gpt-4.1-mini",
    input=f"Summarize the following text in 3-4 sentences:\n{input_text}"
)
gpt_summary = gpt_response.output[0].content[0].text


hf_summary = hf_summarizer(input_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']


embedding1 = similarity_model.encode(gpt_summary, convert_to_tensor=True)
embedding2 = similarity_model.encode(hf_summary, convert_to_tensor=True)
similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item()


print("\n--- INPUT TEXT ---")
print(input_text)

print("\n--- GPT SUMMARY ---")
print(gpt_summary)

print("\n--- HUGGING FACE SUMMARY ---")
print(hf_summary)

print("\n--- COMPARISON ---")
print(f"Semantic Similarity Score: {similarity_score:.2f}")

if similarity_score > 0.85:
    print("✅ Both summaries convey very similar meaning.")
elif similarity_score > 0.6:
    print("⚠️ Summaries have moderate similarity; review both.")
else:
    print("❌ Summaries differ significantly; manual evaluation needed.")

```

# Sample output:
```

Artificial Intelligence (AI) is transforming industries across the world...

AI is revolutionizing industries like healthcare and transport by improving
efficiency and innovation, but ethical and responsible use are essential.

AI enhances productivity across sectors such as healthcare and transport, 
though ethical and responsible usage is key to its sustainability.

Semantic Similarity Score: 0.89
✅ Both summaries convey very similar meaning.

```
# Analysis :

- Both models produced semantically similar summaries.

- The Hugging Face model tends to be more concise.

- The GPT model captures nuanced phrasing and tone better.

- The similarity score (≈0.89) indicates strong alignment.
This experiment demonstrates that Python can orchestrate multiple AI tools for combined insights, useful in research, analytics, or intelligent automation systems.


# Conclusion:

A Python program integrating multiple AI tools was successfully developed and executed. The system effectively compared and analyzed AI outputs, demonstrating interoperability and comparative analysis between OpenAI GPT and Hugging Face Transformers.




# Result: 
The corresponding Prompt is executed successfully.
