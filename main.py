from flask import Flask, request, jsonify, render_template
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)

GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama3-8b-8192"
GOOGLE_DOC_URL = "https://docs.google.com/document/d/196veS3lJcHJ7iJDSN47nnWO9XKHVoxBrSwtSCD8lvUM/edit?usp=sharing"

def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang in ['ur', 'hi']:
            return GoogleTranslator(source='auto', target='en').translate(text)
        return text
    except:
        return text

def get_text_from_google_doc(doc_url):
    doc_id = doc_url.split("/d/")[1].split("/")[0]
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    response = requests.get(export_url)
    return response.text

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def get_relevant_chunks_tfidf(query, docs, k=3):
    vectorizer = TfidfVectorizer().fit(docs + [query])
    vectors = vectorizer.transform(docs + [query])
    sims = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_k = sims.argsort()[-k:][::-1]
    return [docs[i] for i in top_k]

def get_relevant_chunks_smart(query, docs, k=3):
    keyword_matches = [doc for doc in docs if any(word.lower() in doc.lower() for word in query.split())]
    if keyword_matches:
        return keyword_matches[:k]
    return get_relevant_chunks_tfidf(query, docs, k)

def generate_response(query, docs):
    query = translate_to_english(query)
    context_chunks = get_relevant_chunks_smart(query, docs)
    context = "\n".join(context_chunks)

    prompt = f"""You are a highly intelligent CRM assistant. Use the following document context to answer the customer's question precisely. 

Context:
"""
{context}
"""

Question: {query}

Instructions:
- Provide a short, clear, professional response (1–2 sentences)
- Answer **only** from the above context
- Do **not** mention the document or repeat the question
- If answer is not availbale request customer to contact info@soopercart.com

Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful CRM assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

documents = chunk_text(get_text_from_google_doc(GOOGLE_DOC_URL))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response = generate_response(user_input, documents)
    return jsonify({"response": response})

@app.route('/ping')
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)