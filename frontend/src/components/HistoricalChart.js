import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-black/85 backdrop-blur-xl border border-white/10 rounded-sm px-3 py-2">
      <p className="text-xs text-[#A1A1AA] mb-1">Point {label}</p>
      <p className="mono text-sm font-semibold text-white">{payload[0].value.toFixed(2)} min</p>
    </div>
  );
};

const HistoricalChart = ({ data }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card-surface p-6" data-testid="historical-chart">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={16} className="text-[#007AFF]" strokeWidth={1.5} />
          <h2 className="text-lg font-bold tracking-tight">Historical Delay Trends</h2>
        </div>
        <div className="h-[200px] flex items-center justify-center text-[#A1A1AA] text-sm">
          No historical data available
        </div>
      </div>
    );
  }

  const chartData = data.slice(-150).map((item, i) => ({
    index: i,
    delay: parseFloat(item.delay_minutes || 0)
  }));

  const avg = chartData.reduce((s, d) => s + d.delay, 0) / chartData.length;
  const min = Math.min(...chartData.map(d => d.delay));
  const max = Math.max(...chartData.map(d => d.delay));

  return (
    <div className="card-surface p-6 animate-in stagger-5" data-testid="historical-chart">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp size={16} className="text-[#007AFF]" strokeWidth={1.5} />
          <h2 className="text-lg font-bold tracking-tight">Historical Delay Trends</h2>
        </div>
        <div className="flex items-center gap-4">
          <MiniStat label="AVG" value={`${avg.toFixed(1)}m`} color="#007AFF" />
          <MiniStat label="MIN" value={`${min.toFixed(1)}m`} color="#34C759" />
          <MiniStat label="MAX" value={`${max.toFixed(1)}m`} color="#FF3B30" />
        </div>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="delayGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#007AFF" stopOpacity={0.3} />
              <stop offset="100%" stopColor="#007AFF" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="index"
            stroke="#2A2A2A"
            tick={{ fontSize: 10, fill: '#A1A1AA' }}
            axisLine={{ stroke: '#2A2A2A' }}
            tickLine={false}
          />
          <YAxis
            stroke="#2A2A2A"
            tick={{ fontSize: 10, fill: '#A1A1AA', fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
            width={40}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="delay"
            stroke="#007AFF"
            strokeWidth={1.5}
            fill="url(#delayGrad)"
            animationDuration={1200}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

const MiniStat = ({ label, value, color }) => (
  <div className="text-right">
    <div className="label-upper text-[9px]">{label}</div>
    <div className="mono text-sm font-semibold" style={{ color }}>{value}</div>
  </div>
);

export default HistoricalChart;
