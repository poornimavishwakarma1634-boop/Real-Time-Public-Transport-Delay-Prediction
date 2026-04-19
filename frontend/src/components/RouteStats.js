import React from 'react';
import { Bus, Train } from 'lucide-react';

const RouteStats = ({ stats }) => {
  if (!stats || stats.length === 0) return null;

  return (
    <div className="card-surface p-6 animate-in" data-testid="route-stats">
      <h2 className="text-lg font-bold tracking-tight mb-4">Route Performance Statistics</h2>
      <div className="overflow-x-auto">
        <table className="w-full" data-testid="route-stats-table">
          <thead>
            <tr className="border-b border-[#2A2A2A]">
              <th className="label-upper text-left py-3 px-3 text-[10px]">Route</th>
              <th className="label-upper text-left py-3 px-3 text-[10px]">City</th>
              <th className="label-upper text-center py-3 px-3 text-[10px]">Type</th>
              <th className="label-upper text-right py-3 px-3 text-[10px]">Avg Delay</th>
              <th className="label-upper text-right py-3 px-3 text-[10px]">Std Dev</th>
              <th className="label-upper text-right py-3 px-3 text-[10px]">Min</th>
              <th className="label-upper text-right py-3 px-3 text-[10px]">Max</th>
              <th className="label-upper text-right py-3 px-3 text-[10px]">Samples</th>
            </tr>
          </thead>
          <tbody>
            {stats.sort((a, b) => a.avg_delay - b.avg_delay).map((row, idx) => (
              <tr key={idx} className="border-b border-[#2A2A2A]/50 hover:bg-[#1E1E1E] transition-colors">
                <td className="py-2.5 px-3 text-sm font-medium">{row.route_name}</td>
                <td className="py-2.5 px-3 text-sm text-[#A1A1AA]">{row.city}</td>
                <td className="py-2.5 px-3 text-center">
                  <span className="inline-flex items-center gap-1.5 text-xs">
                    {row.transport_type === 'Bus' ? (
                      <Bus size={12} className="text-[#FFCC00]" strokeWidth={1.5} />
                    ) : (
                      <Train size={12} className="text-[#007AFF]" strokeWidth={1.5} />
                    )}
                    {row.transport_type}
                  </span>
                </td>
                <td className="py-2.5 px-3 text-right">
                  <span className={`mono text-sm font-semibold ${
                    row.avg_delay < 5 ? 'text-[#34C759]' :
                    row.avg_delay < 10 ? 'text-[#FFCC00]' : 'text-[#FF3B30]'
                  }`}>
                    {row.avg_delay?.toFixed(1)}m
                  </span>
                </td>
                <td className="py-2.5 px-3 text-right mono text-sm text-[#A1A1AA]">{row.std_delay?.toFixed(1)}</td>
                <td className="py-2.5 px-3 text-right mono text-sm text-[#34C759]">{row.min_delay?.toFixed(1)}m</td>
                <td className="py-2.5 px-3 text-right mono text-sm text-[#FF3B30]">{row.max_delay?.toFixed(1)}m</td>
                <td className="py-2.5 px-3 text-right mono text-sm text-[#A1A1AA]">{row.count?.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default RouteStats;
