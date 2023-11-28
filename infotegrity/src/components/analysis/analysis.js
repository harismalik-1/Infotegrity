import React from 'react';
import { useLocation } from 'react-router-dom';
import "./analysis.css";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
  } from 'chart.js';
  import { Bar } from 'react-chartjs-2';
  
  ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
  );


const Analysis = () => {
  const location = useLocation();
  const analysisResults = location.state?.analysisResults;

  if (!analysisResults) {
    return <p>No analysis data available.</p>;
  }
  const sentimentLabels = Object.keys(analysisResults.roBERTaScores || {});
  const sentimentValues = Object.values(analysisResults.roBERTaScores || {});


  const sentimentData = {
    labels: sentimentLabels,
    datasets: [
      {
        label: 'Sentiment Scores',
        data: sentimentValues,
        backgroundColor: ['rgba(54, 162, 235, 0.6)'],
        borderColor: ['rgba(54, 162, 235, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const textBlobLabels = analysisResults.textBlobAnalysis?.map((_, index) => `Sentence ${index + 1}`) || [];
const polarityData = analysisResults.textBlobAnalysis?.map(analysis => analysis.polarity) || [];
const subjectivityData = analysisResults.textBlobAnalysis?.map(analysis => analysis.subjectivity) || [];

const textBlobChartData = {
  labels: textBlobLabels,
  datasets: [
    {
      label: 'Polarity',
      data: polarityData,
      borderColor: 'rgba(255, 99, 132, 1)',
      backgroundColor: 'rgba(255, 99, 132, 0.2)',
    },
    {
      label: 'Subjectivity',
      data: subjectivityData,
      borderColor: 'rgba(54, 162, 235, 1)',
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
    },
  ],
};


  return (
    <div className="analysis-container">
      <h2>Analysis Results</h2>
      <h3>Article Title: {analysisResults.articleTitle}</h3>

      <section className="overall-bias">
        <h4>Overall Political Bias Assessment</h4>
        <p>{analysisResults.dominantPoliticalBias}</p>
      </section>

      <section>
        <h4>Political Bias</h4>
        {analysisResults.politicalBias?.map((bias, index) => (
          <p key={index}>{bias}</p>
        ))}
      </section>

      <section>
        <h4>Sentiment Analysis (RoBERTa Model)</h4>
        <Bar data={sentimentData} />
      </section>

      <section>
        <h4>TextBlob Sentiment Analysis</h4>
        <Bar data={textBlobChartData} />
      </section>

      <section>
        <h4>Named Entities</h4>
        {analysisResults.namedEntities?.map((entity, index) => (
          <p key={index}>{entity}</p>
        ))}
      </section>
    </div>
  );
};

export default Analysis;
