import React, { useRef, useEffect, useState } from "react";
import { Camera, RefreshCw, Upload, AlertCircle } from "lucide-react";

export const CameraView = ({ onCapture, frozenPreview }) => {
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);
  const [facingMode, setFacingMode] = useState("user"); // Default to 'user' for laptop compatibility
  const [cameraError, setCameraError] = useState(null);

  useEffect(() => {
    if (frozenPreview) return;

    let currentStream = null;

    async function startCamera() {
      setCameraError(null);

      // Check if browser supports camera API
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraError(
          "Camera API is not supported. Please use http://localhost:5173",
        );
        return;
      }

      try {
        // Attempt 1: Try with preferred facingMode
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: facingMode },
            audio: false,
          });
        } catch (constraintErr) {
          // Attempt 2: Fallback for laptops that reject 'facingMode'
          console.warn(
            "FacingMode rejected, falling back to basic video...",
            constraintErr,
          );
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current
              .play()
              .catch((e) => console.log("Play interrupted:", e));
          };
        }
        currentStream = stream;
      } catch (err) {
        console.error("Camera access error:", err);
        if (
          err.name === "NotAllowedError" ||
          err.name === "PermissionDeniedError"
        ) {
          setCameraError(
            "Camera permission denied. Please allow camera access in your browser settings.",
          );
        } else if (err.name === "NotReadableError") {
          setCameraError(
            "Camera is in use by another app (e.g. Zoom, Teams, or Windows Hello).",
          );
        } else if (err.name === "NotFoundError") {
          setCameraError("No webcam found on this device.");
        } else {
          setCameraError(`Camera error: ${err.message || err.name}`);
        }
      }
    }

    startCamera();

    return () => {
      if (currentStream) {
        currentStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [facingMode, frozenPreview]);

  const handleCapture = () => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (blob) onCapture(blob);
      },
      "image/jpeg",
      0.92,
    );
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onCapture(file);
    }
  };

  const toggleCamera = () => {
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
  };

  return (
    <div className="relative w-full h-full bg-black flex items-center justify-center overflow-hidden">
      {/* Hidden File Input for desktop testing */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileUpload}
        accept="image/*"
        className="hidden"
      />

      {frozenPreview ? (
        <img
          src={frozenPreview}
          alt="Captured Scene"
          className="w-full h-full object-cover"
        />
      ) : (
        <>
          {cameraError ? (
            <div className="flex flex-col items-center justify-center p-6 text-center max-w-sm">
              <div className="bg-red-500/20 p-4 rounded-full mb-4 border border-red-500/30">
                <AlertCircle className="w-8 h-8 text-red-400" />
              </div>
              <p className="text-sm font-medium text-red-200 mb-6">
                {cameraError}
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-blue-600 hover:bg-blue-500 text-white px-5 py-3 rounded-xl flex items-center gap-2 text-sm font-medium shadow-lg transition-all"
              >
                <Upload className="w-4 h-4" /> Upload Photo from Disk
              </button>
            </div>
          ) : (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
          )}

          {/* Controls Overlay */}
          {!cameraError && (
            <div className="absolute bottom-8 inset-x-0 flex items-center justify-around px-8 z-30">
              {/* Flip camera / toggle */}
              <button
                onClick={toggleCamera}
                className="bg-black/40 backdrop-blur-md p-3.5 rounded-full border border-white/20 text-white hover:bg-black/60 transition-all active:scale-95"
                title="Switch Camera"
              >
                <RefreshCw className="w-5 h-5" />
              </button>

              {/* Shutter Button */}
              <button
                onClick={handleCapture}
                className="w-20 h-20 rounded-full border-4 border-white p-1.5 flex items-center justify-center transition-all active:scale-90"
                title="Capture Frame"
              >
                <div className="w-full h-full bg-white rounded-full flex items-center justify-center shadow-lg">
                  <Camera className="w-6 h-6 text-black" />
                </div>
              </button>

              {/* Upload image file from laptop */}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-black/40 backdrop-blur-md p-3.5 rounded-full border border-white/20 text-white hover:bg-black/60 transition-all active:scale-95"
                title="Upload Photo"
              >
                <Upload className="w-5 h-5" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};
