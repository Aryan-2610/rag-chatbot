from flask import Flask, request, jsonify, render_template
from rag_model import get_rag_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "")
    response = get_rag_response(query)
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)
