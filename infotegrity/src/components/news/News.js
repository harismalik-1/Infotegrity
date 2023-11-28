import React, { useEffect, useState } from "react";
import { useNavigate } from 'react-router-dom';
import "./News.css";
import Analysis from "../analysis/analysis";


const News = () => {
  const [mynews, setMyNews] = useState([]);
  const [articleLink, setArticleLink] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const navigate = useNavigate();



  const fetchData = async () => {
    let response = await fetch(
      "https://newsapi.org/v2/top-headlines?country=us&apiKey=714ef9b8a6ef47d19b4bda6f4f0d100f"
      );
    let data = await response.json();
    setMyNews(data.articles);
  };

  useEffect(() => {
    fetchData();
  }, []);


  const handleAnalyzeClick = async () => {
    if (!articleLink) {
      alert('Please enter a URL to analyze.');
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', { //Flask server running on localhost:5000
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: articleLink }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const result = await response.json();
      navigate('/analyze', { state: { analysisResults: result } });
    } catch (error) {
      console.error('Error fetching analysis:', error);
      alert('Failed to fetch analysis results.');
    }
  };

  return (
    <>
      <h1 className="text-center my-3">Analyze any News Article!</h1>
      <div className="analyze-bar">
        <input 
          type="text" 
          className="article-input" 
          placeholder="Paste the link of the article here"
          value={articleLink}
          onChange={(e) => setArticleLink(e.target.value)}
        />
        <button className="btn analyze-btn" onClick={handleAnalyzeClick}>Analyze</button>
      </div>
      <div className="mainDiv">
        {mynews.map((ele, index) => (
          <div key={index} className="card">
            <div className="card-content">
              <img src={ele.urlToImage || "/default.png"} className="card-img-top" alt="..." />
              <div className="card-body">
                <h5 className="card-title">{ele.author || "News Article"}</h5>
                <p className="card-text">{ele.title}</p>
              </div>
            </div>
            <div className="card-footer">
              <a href={ele.url} target="_blank" rel="noopener noreferrer" className="btn btn-primary mr-2">
                Read More
              </a>
              <a className="btn btn-primary" onClick={handleAnalyzeClick}>
                Analyze
              </a>
            </div>
          </div>
        ))}
      </div>
      {analysisResults && <Analysis analysisResults={analysisResults} />}
    </>
  );
};

export default News;
