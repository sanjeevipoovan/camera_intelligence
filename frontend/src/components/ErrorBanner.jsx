import React from "react";
import { AlertTriangle, X } from "lucide-react";

export const ErrorBanner = ({ message, onDismiss }) => {
  return (
    <div className="absolute top-4 left-4 right-4 z-50 bg-red-950/90 border border-red-500/50 backdrop-blur-md rounded-2xl p-4 flex items-start gap-3 shadow-2xl animate-in slide-in-from-top-4 duration-200">
      <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      <div className="flex-1 text-sm text-red-200 font-medium leading-relaxed">
        {message}
      </div>
      <button onClick={onDismiss} className="text-red-400 hover:text-white p-1">
        <X className="w-4 h-4" />
      </button>
    </div>
  );
};
