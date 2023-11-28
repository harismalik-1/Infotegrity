import React, { useEffect, useState } from "react";
import "./News.css";

const News = () => {
  const [mynews, setMyNews] = useState([]);
  const [articleLink, setArticleLink] = useState('');

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
        <button className="btn analyze-btn" >Analyze</button>
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
              <a className="btn btn-primary">
                Analyze
              </a>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};

export default News;
