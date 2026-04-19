import React from 'react';
import { Clock, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const PredictionDisplay = ({ prediction, loading }) => {
  if (!prediction && !loading) {
    return (
      <div className="card-surface p-8 h-full flex flex-col items-center justify-center" data-testid="prediction-display">
        <div className="text-center">
          <Clock size={48} className="text-[#2A2A2A] mx-auto mb-4" strokeWidth={1} />
          <h3 className="text-lg font-bold tracking-tight text-[#A1A1AA] mb-1">Awaiting Prediction</h3>
          <p className="text-sm text-[#A1A1AA]/60">Select a route and time, then click Predict Delay</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="card-surface p-8 h-full flex flex-col items-center justify-center" data-testid="prediction-loading">
        <div className="text-center">
          <div className="inline-block w-12 h-12 border-2 border-[#2A2A2A] border-t-[#007AFF] rounded-full animate-spin mb-4" />
          <h3 className="text-lg font-bold tracking-tight mb-1">Analyzing Patterns...</h3>
          <p className="text-sm text-[#A1A1AA]">Running ML model inference</p>
        </div>
      </div>
    );
  }

  const delay = prediction.predicted_delay;
  const getDelayLevel = (d) => {
    if (d < 3) return { color: '#34C759', label: 'ON TIME', icon: CheckCircle, bg: 'rgba(52,199,89,0.08)' };
    if (d < 7) return { color: '#FFCC00', label: 'MINOR DELAY', icon: AlertTriangle, bg: 'rgba(255,204,0,0.08)' };
    return { color: '#FF3B30', label: 'SIGNIFICANT DELAY', icon: XCircle, bg: 'rgba(255,59,48,0.08)' };
  };

  const level = getDelayLevel(delay);
  const Icon = level.icon;

  return (
    <div className="card-surface p-6 h-full flash-update" data-testid="prediction-result">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-bold tracking-tight">Prediction Result</h2>
        <div className="flex items-center gap-2">
          <Icon size={14} style={{ color: level.color }} strokeWidth={1.5} />
          <span className="label-upper" style={{ color: level.color }}>{level.label}</span>
        </div>
      </div>

      {/* Main Prediction */}
      <div className="rounded-sm p-8 mb-6 text-center" style={{ backgroundColor: level.bg, border: `1px solid ${level.color}20` }}>
        <div className="metric-value text-6xl md:text-7xl mb-2" style={{ color: level.color }} data-testid="predicted-delay">
          {delay}
        </div>
        <div className="text-sm text-[#A1A1AA] tracking-wide">MINUTES DELAY</div>
      </div>

      {/* Meta Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetaCell label="MODEL" value={prediction.model_used} testId="model-used" />
        <MetaCell label="CONFIDENCE" value={prediction.confidence} testId="confidence" />
        <MetaCell label="ROUTE" value={prediction.route_id} testId="route-id" />
        <MetaCell label="TIME" value={new Date(prediction.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} testId="prediction-time" />
      </div>
    </div>
  );
};

const MetaCell = ({ label, value, testId }) => (
  <div className="p-3 bg-[#1E1E1E] border border-[#2A2A2A] rounded-sm">
    <div className="label-upper text-[10px] mb-1">{label}</div>
    <div className="mono text-sm font-semibold capitalize truncate" data-testid={testId}>{value}</div>
  </div>
);

export default PredictionDisplay;
