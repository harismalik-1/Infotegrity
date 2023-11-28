import React from 'react'
import Navbar from './components/navbar/Navbar'
import News from './components/news/News'
import Analysis from './components/analysis/analysis'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/layout';

const App = () => {
  return (
    <Router>
      <Routes>
      <Route path="/" element={<Layout />}>
          <Route index element={<News />} />
          <Route path="analyze" element={<Analysis />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App
