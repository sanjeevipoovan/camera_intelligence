import React, { useState } from "react";
import { CameraView } from "./components/CameraView";
import { ResponseCard } from "./components/ResponseCard";
import { ErrorBanner } from "./components/ErrorBanner";
import { analyzeImage } from "./services/api";
import { Sparkles, Loader2 } from "lucide-react";

export default function App() {
  const [capturedBlob, setCapturedBlob] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleCapture = async (blob) => {
    setCapturedBlob(blob);
    setPreviewUrl(URL.createObjectURL(blob));
    setErrorMessage(null);

    await submitQuery(
      blob,
      "Describe what is visible in this image and highlight any notable details.",
      null,
    );
  };

  const submitQuery = async (blob, questionText, sessionId) => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const res = await analyzeImage(blob, questionText, sessionId);
      setResponse(res);
    } catch (err) {
      setErrorMessage(err.message || "Failed to process photo.");
      if (!sessionId) {
        setPreviewUrl(null);
        setCapturedBlob(null);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleFollowUp = (question) => {
    if (!response) return;
    submitQuery(null, question, response.session_id);
  };

  const handleReset = () => {
    setCapturedBlob(null);
    setPreviewUrl(null);
    setResponse(null);
    setErrorMessage(null);
  };

  return (
    <div className="relative w-full h-[100dvh] overflow-hidden bg-black font-sans">
      {/* Top Header */}
      <div className="absolute top-0 inset-x-0 z-30 flex items-center justify-between px-4 pt-[max(1rem,env(safe-area-inset-top))] pb-4 bg-gradient-to-b from-black/80 to-transparent">
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <span className="font-bold text-sm sm:text-base tracking-wide text-white">
            Visual Intelligence
          </span>
        </div>
      </div>

      {/* Error Alert */}
      {errorMessage && (
        <ErrorBanner
          message={errorMessage}
          onDismiss={() => setErrorMessage(null)}
        />
      )}

      {/* Main Viewfinder */}
      <CameraView onCapture={handleCapture} frozenPreview={previewUrl} />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="absolute inset-0 z-50 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center gap-3 px-6 text-center">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          <p className="text-sm font-medium text-white/90">
            Analyzing scene with LiquidAI...
          </p>
        </div>
      )}

      {/* Output Card */}
      {response && (
        <ResponseCard
          response={response}
          onFollowUp={handleFollowUp}
          onReset={handleReset}
          isLoading={isLoading}
        />
      )}
    </div>
  );
}
