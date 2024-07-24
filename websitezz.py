from flask import Flask, request, render_template
from backend_model import generate_short_info, generate_paragraph, get_youtube_videos, get_books, get_research_papers, get_suggestions

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    try:
        summary = generate_short_info(query)
        paragraph = generate_paragraph(query)
        youtube_results = get_youtube_videos(query)
        books_results = get_books(query)
        papers_results = get_research_papers(query)
        suggestions = get_suggestions(query)
        return render_template('results.html', query=query, summary=summary, paragraph=paragraph, 
                               youtube=youtube_results, books=books_results, papers=papers_results, 
                               suggestions=suggestions)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)