from flask import Flask, request, jsonify
from flask_cors import CORS
import sentiment_analysis_module

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    url = data['url']
    result = sentiment_analysis_module.analyze_url(url)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
