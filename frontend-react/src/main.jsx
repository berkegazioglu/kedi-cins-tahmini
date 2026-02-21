import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.jsx'
import CatTranslator from './pages/CatTranslator.jsx'
import ResultPage from './pages/ResultPage.jsx'
import './styles/index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/cat-translator" element={<CatTranslator />} />
        <Route path="/result" element={<ResultPage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
