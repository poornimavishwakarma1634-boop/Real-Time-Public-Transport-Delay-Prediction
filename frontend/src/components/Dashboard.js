import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { BarChart3, Train, Activity, Database, Cpu, MapPin, Radio, RefreshCw } from 'lucide-react';
import RouteSelector from './RouteSelector';
import PredictionDisplay from './PredictionDisplay';
import HistoricalChart from './HistoricalChart';
import RouteStats from './RouteStats';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const StatCard = ({ label, value, icon: Icon, color, delay, sub }) => (
  <div className={`card-surface p-5 animate-in stagger-${delay}`} data-testid={`stat-${label.toLowerCase().replace(/\s/g, '-')}`}>
    <div className="flex items-center justify-between mb-3">
      <span className="label-upper">{label}</span>
      <Icon size={16} style={{ color }} strokeWidth={1.5} />
    </div>
    <div className="metric-value text-3xl" style={{ color }}>{value}</div>
    {sub && <div className="text-xs text-[#A1A1AA] mt-1">{sub}</div>}
  </div>
);

const Dashboard = () => {
  const [routes, setRoutes] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedRoute, setSelectedRoute] = useState('BLR-BUS-1');
  const [selectedModel, setSelectedModel] = useState('PROPHET');
  const [selectedTime, setSelectedTime] = useState(new Date().toISOString());
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historicalData, setHistoricalData] = useState([]);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [routeStats, setRouteStats] = useState([]);
  const [gtfsStatus, setGtfsStatus] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      const [routesRes, modelsRes, histRes, statusRes, statsRes, gtfsRes] = await Promise.all([
        axios.get(`${API}/routes`).catch(() => ({ data: { routes: [] } })),
        axios.get(`${API}/models`).catch(() => ({ data: { models: ['PROPHET', 'TENSORFLOW_LSTM', 'PYTORCH_LSTM'] } })),
        axios.get(`${API}/historical?limit=500`).catch(() => ({ data: { data: [] } })),
        axios.get(`${API}/pipeline-status`).catch(() => ({ data: { pipeline_complete: false, models_trained: [], scheduler: { status: 'unknown' } } })),
        axios.get(`${API}/route-stats`).catch(() => ({ data: { stats: [] } })),
        axios.get(`${API}/gtfs-status`).catch(() => ({ data: { gtfs: {}, live_feed_count: 0 } }))
      ]);
      setRoutes(routesRes.data.routes);
      setModels(modelsRes.data.models);
      setHistoricalData(histRes.data.data);
      setPipelineStatus(statusRes.data);
      setRouteStats(statsRes.data.stats);
      setGtfsStatus(gtfsRes.data);
    } catch (e) {
      console.error('Fetch error:', e);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API}/predict`, {
        timestamp: selectedTime,
        route_id: selectedRoute,
        model_name: selectedModel
      });
      setPrediction(res.data);
    } catch (e) {
      console.error('Prediction error:', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      {/* Header */}
      <header className="glass-header sticky top-0 z-50 px-6 py-4" data-testid="dashboard-header">
        <div className="max-w-[1440px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Train size={24} className="text-[#007AFF]" strokeWidth={1.5} />
              <h1 className="text-xl font-bold tracking-tight" data-testid="dashboard-title">
                TransitML
              </h1>
            </div>
            <span className="text-xs text-[#A1A1AA] border-l border-[#2A2A2A] pl-4">
              Bangalore & Kalaburagi
            </span>
          </div>
          <div className="flex items-center gap-4">
            {/* Pipeline Status */}
            <div className="flex items-center gap-2" data-testid="pipeline-status">
              <div className={`w-2 h-2 rounded-full ${pipelineStatus?.scheduler?.status === 'running' ? 'bg-[#34C759] live-dot' : 'bg-[#FF3B30]'}`} />
              <span className="text-xs text-[#A1A1AA]">
                {pipelineStatus?.scheduler?.status === 'running' ? 'Live — Auto-retraining' : 'Initializing...'}
              </span>
            </div>
            <Link
              to="/comparison"
              className="flex items-center gap-2 px-4 py-2 bg-[#141414] border border-[#2A2A2A] rounded-sm text-sm hover:bg-[#1E1E1E] hover:border-[#3A3A3A] transition-all"
              data-testid="view-comparison-btn"
            >
              <BarChart3 size={16} strokeWidth={1.5} />
              Model Comparison
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-[1440px] mx-auto p-4 md:p-6 space-y-6">
        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4" data-testid="stats-row">
          <StatCard label="ML Models" value={models.length || 3} icon={Cpu} color="#007AFF" delay={1} sub="Prophet, TF LSTM, PyTorch" />
          <StatCard label="Active Routes" value={routes.length || 8} icon={MapPin} color="#34C759" delay={2} sub="Bangalore & Kalaburagi" />
          <StatCard label="GTFS Routes" value={gtfsStatus?.gtfs?.routes || '4.2K'} icon={Database} color="#FFCC00" delay={3} sub="BMTC + NEKRTC feeds" />
          <StatCard label="Live Records" value={pipelineStatus?.live_feed_records || 0} icon={Radio} color="#32ADE6" delay={4} sub="Real-time observations" />
          <StatCard label="Scheduler" value={pipelineStatus?.scheduler?.status === 'running' ? 'ACTIVE' : 'INIT'} icon={RefreshCw} color={pipelineStatus?.scheduler?.status === 'running' ? '#34C759' : '#FF3B30'} delay={5} sub={pipelineStatus?.scheduler?.retrain_count > 0 ? `${pipelineStatus.scheduler.retrain_count} retrains` : 'Auto-retrain 60min'} />
        </div>

        {/* Main Grid: Selector + Prediction */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-4">
            <RouteSelector
              routes={routes}
              models={models}
              selectedRoute={selectedRoute}
              selectedModel={selectedModel}
              selectedTime={selectedTime}
              onRouteChange={setSelectedRoute}
              onModelChange={setSelectedModel}
              onTimeChange={setSelectedTime}
              onPredict={handlePredict}
              loading={loading}
            />
          </div>
          <div className="lg:col-span-8">
            <PredictionDisplay prediction={prediction} loading={loading} />
          </div>
        </div>

        {/* Historical Chart */}
        <HistoricalChart data={historicalData} />

        {/* Route Stats */}
        <RouteStats stats={routeStats} />
      </main>

      {/* Footer */}
      <footer className="border-t border-[#2A2A2A] mt-12 py-6 px-6">
        <div className="max-w-[1440px] mx-auto flex items-center justify-between text-xs text-[#A1A1AA]">
          <span>TransitML - Real-Time Public Transport Delay Prediction</span>
          <span>Models: ARIMA, SARIMA, Prophet, TensorFlow LSTM, PyTorch LSTM</span>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;
