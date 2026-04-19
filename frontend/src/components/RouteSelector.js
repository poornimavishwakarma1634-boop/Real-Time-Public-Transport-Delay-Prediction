import React from 'react';
import { Bus, Train, Zap } from 'lucide-react';

const RouteSelector = ({
  routes, models, selectedRoute, selectedModel, selectedTime,
  onRouteChange, onModelChange, onTimeChange, onPredict, loading
}) => {
  const getCurrentDateTime = () => {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    return now.toISOString().slice(0, 16);
  };

  const selectedRouteInfo = routes.find(r => r.id === selectedRoute);

  return (
    <div className="card-surface p-6 h-full flex flex-col" data-testid="route-selector">
      <div className="flex items-center gap-2 mb-6">
        <MapPin size={16} className="text-[#007AFF]" strokeWidth={1.5} />
        <h2 className="text-lg font-bold tracking-tight">Configuration</h2>
      </div>

      {/* City Filter */}
      <div className="mb-5">
        <label className="label-upper block mb-2">Route</label>
        <select
          value={selectedRoute}
          onChange={(e) => onRouteChange(e.target.value)}
          className="w-full px-3 py-2.5 rounded-sm text-sm"
          data-testid="route-select"
        >
          <optgroup label="Bangalore - Bus">
            {routes.filter(r => r.city === 'Bangalore' && r.type === 'Bus').map(r => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </optgroup>
          <optgroup label="Bangalore - Train">
            {routes.filter(r => r.city === 'Bangalore' && r.type === 'Train').map(r => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </optgroup>
          <optgroup label="Kalaburagi - Bus">
            {routes.filter(r => r.city === 'Kalaburagi' && r.type === 'Bus').map(r => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </optgroup>
          <optgroup label="Kalaburagi - Train">
            {routes.filter(r => r.city === 'Kalaburagi' && r.type === 'Train').map(r => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </optgroup>
          <optgroup label="Intercity">
            {routes.filter(r => r.city === 'Intercity').map(r => (
              <option key={r.id} value={r.id}>{r.name}</option>
            ))}
          </optgroup>
        </select>
      </div>

      {/* Route Badge */}
      {selectedRouteInfo && (
        <div className="flex items-center gap-2 mb-5 px-3 py-2 bg-[#1E1E1E] border border-[#2A2A2A] rounded-sm">
          {selectedRouteInfo.type === 'Bus' ? (
            <Bus size={14} className="text-[#FFCC00]" strokeWidth={1.5} />
          ) : (
            <Train size={14} className="text-[#007AFF]" strokeWidth={1.5} />
          )}
          <span className="text-xs text-[#A1A1AA]">{selectedRouteInfo.city}</span>
          <span className="text-xs px-1.5 py-0.5 bg-[#2A2A2A] rounded-sm text-[#A1A1AA]">
            {selectedRouteInfo.type}
          </span>
        </div>
      )}

      {/* Model */}
      <div className="mb-5">
        <label className="label-upper block mb-2">Model</label>
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          className="w-full px-3 py-2.5 rounded-sm text-sm"
          data-testid="model-select"
        >
          {(models.length > 0 ? models : ['ARIMA', 'SARIMA', 'Prophet']).map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* Time */}
      <div className="mb-6">
        <label className="label-upper block mb-2">Prediction Time</label>
        <input
          type="datetime-local"
          defaultValue={getCurrentDateTime()}
          onChange={(e) => onTimeChange(new Date(e.target.value).toISOString())}
          className="w-full px-3 py-2.5 rounded-sm text-sm"
          data-testid="time-input"
        />
      </div>

      {/* Predict */}
      <button
        onClick={onPredict}
        disabled={loading}
        className={`w-full py-3 rounded-sm font-semibold text-sm transition-all flex items-center justify-center gap-2 ${
          loading
            ? 'bg-[#2A2A2A] text-[#A1A1AA] cursor-not-allowed'
            : 'bg-[#007AFF] text-white hover:bg-[#0066DD] active:scale-[0.98]'
        }`}
        data-testid="predict-btn"
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Processing...
          </span>
        ) : (
          <>
            <Zap size={16} strokeWidth={1.5} />
            Predict Delay
          </>
        )}
      </button>

      {/* Info */}
      <div className="mt-auto pt-5 border-t border-[#2A2A2A] mt-5">
        <p className="text-xs text-[#A1A1AA] leading-relaxed">
          Predictions powered by 5 ML models analyzing historical patterns, rush hours, weather, and route characteristics for Bangalore & Kalaburagi transport networks.
        </p>
      </div>
    </div>
  );
};

const MapPin = ({ size, className, strokeWidth }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" className={className}>
    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
    <circle cx="12" cy="10" r="3"/>
  </svg>
);

export default RouteSelector;
