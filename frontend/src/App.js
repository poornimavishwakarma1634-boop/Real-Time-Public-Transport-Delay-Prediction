import React from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import ModelComparison from './components/ModelComparison';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/comparison" element={<ModelComparison />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
