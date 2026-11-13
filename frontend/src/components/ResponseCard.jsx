import React, { useState } from "react";
import { Sparkles, Send, RotateCcw, Zap } from "lucide-react";

export const ResponseCard = ({ response, onFollowUp, onReset, isLoading }) => {
  const [followUpText, setFollowUpText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!followUpText.trim() || isLoading) return;
    onFollowUp(followUpText.trim());
    setFollowUpText("");
  };

  return (
    <div className="absolute inset-x-0 bottom-0 z-40 bg-surface/95 backdrop-blur-xl border-t border-white/10 rounded-t-3xl p-5 max-h-[75vh] flex flex-col shadow-2xl animate-in slide-in-from-bottom duration-300">
      {/* Header Telemetry Badge */}
      <div className="flex items-center justify-between pb-3 border-b border-white/10 text-xs text-neutral-400">
        <div className="flex items-center gap-1.5 text-blue-400 font-semibold">
          <Sparkles className="w-4 h-4" />
          <span>LiquidAI Omni</span>
        </div>
        <div className="flex items-center gap-2">
          {response.cache_hit && (
            <span className="bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full flex items-center gap-1 font-medium">
              <Zap className="w-3 h-3" /> KV Cache Hit
            </span>
          )}
          <span>
            {response.metrics?.tokens_per_second?.toFixed(1) || 0} t/s
          </span>
          <span>•</span>
          <span>{response.metrics?.inference_time_ms?.toFixed(0) || 0} ms</span>
        </div>
      </div>

      {/* AI Answer Content */}
      <div className="py-4 overflow-y-auto text-sm text-neutral-200 leading-relaxed max-h-56">
        {response.answer}
      </div>

      {/* Follow-up Question Form */}
      <form onSubmit={handleSubmit} className="pt-2 flex items-center gap-2">
        <input
          type="text"
          value={followUpText}
          onChange={(e) => setFollowUpText(e.target.value)}
          placeholder="Ask a follow-up question..."
          disabled={isLoading}
          className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500 transition-colors"
        />
        <button
          type="submit"
          disabled={isLoading || !followUpText.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white p-3 rounded-xl transition-all shrink-0"
        >
          <Send className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={onReset}
          className="bg-neutral-800 hover:bg-neutral-700 text-neutral-300 p-3 rounded-xl transition-all shrink-0"
          title="Snap a new photo"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
};
