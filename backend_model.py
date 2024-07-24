import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, TFAutoModelForCausalLM
import nltk

# summarization and tokenizer (DistilBERT-based)
summarization_model_name = "sshleifer/distilbart-cnn-12-6"
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
summarizer = pipeline('summarization', model=summarization_model, tokenizer=summarization_tokenizer)

# text generation  and tokenizer (DistilGPT-2)
generation_model_name = "distilgpt2"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = TFAutoModelForCausalLM.from_pretrained(generation_model_name)
generator = pipeline('text-generation', model=generation_model, tokenizer=generation_tokenizer)

# NLTK data
nltk.download('punkt')

# APIs
YOUTUBE_API_KEY = 'AIzaSyCvTqkED4Of2ieMwxbNgYVQrDsDEsA41UI'
GOOGLE_BOOKS_API_URL = 'https://www.googleapis.com/books/v1/volumes'
CROSSREF_API_URL = 'https://api.crossref.org/works'

def generate_short_info(query):
    prompt = f"Summarize the key points about {query} in a concise manner."
    summary = summarizer(prompt, max_length=7, num_return_sequences=1)[0]['summary_text']
    return summary

def generate_paragraph(query):
    prompt = f"Provide a detailed, factual explanation strictly about: {query}"
    generated_text = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.7, top_k=50, do_sample=True)[0]['generated_text']
    return generated_text.replace(prompt, "").strip()

def get_youtube_videos(query):
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    videos = response.json().get('items', [])
    return [{'title': video['snippet']['title'], 'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}"} for video in videos]

def get_books(query):
    params = {'q': query}
    response = requests.get(GOOGLE_BOOKS_API_URL, params=params)
    books = response.json().get('items', [])
    return [{'title': book['volumeInfo']['title'], 'thumbnail': book['volumeInfo'].get('imageLinks', {}).get('thumbnail', ''), 'infoLink': book['volumeInfo']['infoLink']} for book in books]

def get_research_papers(query):
    params = {'query.bibliographic': query}
    response = requests.get(CROSSREF_API_URL, params=params)
    papers = response.json().get('message', {}).get('items', [])
    return [{'title': paper['title'][0], 'url': paper['URL']} for paper in papers]

def get_suggestions(query):
    suggestions = [f"Explore more about {query}", f"Related topic: {query} basics"]
    return suggestions

