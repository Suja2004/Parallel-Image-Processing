import React, { useState, useEffect, useRef } from "react";
import { Upload, Image, Loader2, Check, X, Info } from "lucide-react";
import { useDropzone } from "react-dropzone";
import ImageModal from "./ImageModal";
import OutputFiles from "./OutputFiles";
import icon from "./assets/Icon.png";
import PresetSelector from "./PresetSelector";

import {
  checkBackendHealth,
  fetchPresets,
  uploadFiles,
  processImagesApi,
  getTaskStatus,
  API_BASE_URL,
} from "./apis/api";

function App() {
  const [backendStatus, setBackendStatus] = useState("checking");
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [presets, setPresets] = useState({});
  const [selectedPreset, setSelectedPreset] = useState("");
  const [processingTasks, setProcessingTasks] = useState([]);
  const [showBanner, setShowBanner] = useState(false);
  const sentinelRef = useRef(null);

  // 🔹 Check backend + presets on load
  useEffect(() => {
    (async () => {
      try {
        const res = await checkBackendHealth();
        if (res.status === 200) setBackendStatus("connected");
      } catch {
        setBackendStatus("disconnected");
      }

      try {
        const res = await fetchPresets();
        setPresets(res.data);
      } catch (err) {
        console.error("Failed to fetch presets:", err);
      }
    })();
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        setShowBanner(entry.isIntersecting);
      },
      { root: null, threshold: 1.0 }
    );

    if (sentinelRef.current) {
      observer.observe(sentinelRef.current);
    }

    return () => {
      if (sentinelRef.current) observer.unobserve(sentinelRef.current);
    };
  }, []);

  // 🔹 File upload
  const onDrop = async (acceptedFiles) => {
    try {
      const res = await uploadFiles(acceptedFiles);
      setUploadedFiles((prev) => [...prev, ...res.data.uploaded_files]);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed. Please try again.");
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"] },
    maxFiles: 10,
  });

  // 🔹 Process images
  const processImages = async () => {
    if (!selectedPreset || uploadedFiles.length === 0) {
      alert("Please select a preset and upload at least one image");
      return;
    }

    try {
      const payload =
        uploadedFiles.length === 1
          ? { filename: uploadedFiles[0].filename, preset: selectedPreset }
          : { filenames: uploadedFiles.map((f) => f.filename), preset: selectedPreset };

      const res = await processImagesApi(payload, uploadedFiles.length === 1);

      const newTask = {
        ...res.data,
        startTime: Date.now(),
        outputFiles: [],
        status: "pending",
      };

      setProcessingTasks((prev) => [...prev, newTask]);
      pollTaskStatus(newTask.task_id);
    } catch (err) {
      console.error("Processing failed:", err);
      alert("Processing failed. Please try again.");
    }
  };

  // 🔹 Polling task status
  const pollTaskStatus = (taskId) => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await getTaskStatus(taskId);
        const taskData = res.data;

        setProcessingTasks((prev) =>
          prev.map((task) =>
            task.task_id === taskId ? { ...task, ...taskData } : task
          )
        );

        if (taskData.status === "completed" || taskData.status === "failed") {
          clearInterval(pollInterval);
        }
      } catch (err) {
        console.error("Status check failed:", err);
        clearInterval(pollInterval);
      }
    }, 1000);
  };

  const StatusIndicator = () => (
    <div className="flex items-center gap-2 mb-6">
      {backendStatus === 'checking' && (
        <>
          <Loader2 className="w-4 h-4 animate-spin text-yellow-500" />
          <span className="text-yellow-600">Connecting...</span>
        </>
      )}
      {backendStatus === 'connected' && (
        <>
          <Check className="w-4 h-4 text-green-500" />
          <span className="text-green-600">Online</span>
        </>
      )}
      {backendStatus === 'disconnected' && (
        <>
          <X className="w-4 h-4 text-red-500" />
          <span className="text-red-600">Offline – please try again</span>
        </>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 pb-20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="flex items-center justify-center text-4xl font-bold text-gray-800 mb-2">
              <img src={icon} alt="ImagoLab Logo" className="w-10 h-10 mr-2" />
              ImagoLab
            </h1>
            <p className="text-gray-600">
              Upload images and apply multiple filters with advanced processing
            </p>
          </div>

          <StatusIndicator />

          {backendStatus === 'connected' && (
            <div className="grid grid-cols-1 lg:flex justify-around gap-8">
              {/* Upload Section */}
              <div className="card">
                <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                  <Upload className="w-6 h-6 text-primary-600" />
                  Upload Images
                </h2>

                <div
                  {...getRootProps()}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-secondary-300 hover:border-primary-400'
                    }`}
                >
                  <input {...getInputProps()} />
                  <Image className="w-12 h-12 text-secondary-400 mx-auto mb-4" />
                  {isDragActive ? (
                    <p className="text-primary-600">Drop the images here...</p>
                  ) : (
                    <div>
                      <p className="text-secondary-600 mb-2">
                        Drag & drop images here, or click to select
                      </p>
                      <p className="text-sm text-secondary-500">
                        Supports: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
                      </p>
                    </div>
                  )}
                </div>

                {/* Uploaded Files */}
                {uploadedFiles.length > 0 && (
                  <ImageModal files={uploadedFiles} />
                )}

              </div>

              {/* Processing Section */}
              <div className="card">
                <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                  <Info className="w-6 h-6 text-primary-600" />
                  Processing Options
                </h2>

                {/* Preset Selection */}
                <PresetSelector
                  presets={presets}
                  selectedPreset={selectedPreset}
                  setSelectedPreset={setSelectedPreset}
                />

                <button
                  onClick={processImages}
                  disabled={!selectedPreset || uploadedFiles.length === 0}
                  className="button"
                >
                  Process Images
                </button>
              </div>
            </div>
          )}

          {processingTasks.length > 0 && (
            <div className="mt-8">
              <h2 className="text-2xl font-semibold mb-4">Processing Tasks</h2>
              <div className="space-y-4">
                {processingTasks.map((task) => (
                  <div key={task.task_id} className="card">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-medium">
                        Task: {task.batch_type}
                        {task.filters_count && ` (${task.filters_count} filters)`}
                      </h3>
                      <span className={`px-3 py-1 rounded-full text-sm ${task.status === 'completed' ? 'bg-green-100 text-green-700' :
                        task.status === 'failed' ? 'bg-red-100 text-red-700' :
                          'bg-yellow-100 text-yellow-700'
                        }`}>
                        {task.status}
                      </span>
                    </div>

                    <div className="progress-bar mb-3">
                      <div
                        className="progress-fill"
                        style={{ width: `${task.progress || 0}%` }}
                      ></div>
                    </div>
                    <p className="text-sm text-secondary-600 mb-3">
                      Progress: {task.progress || 0}%
                    </p>

                    {task.output_files && task.output_files.length > 0 && (
                      <OutputFiles task={task} API_BASE_URL={API_BASE_URL} />
                    )}


                    {task.error_message && (
                      <p className="text-red-600 text-sm mt-2">{task.error_message}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <div ref={sentinelRef} className="h-1"></div>

      <div
        className={`fixed bottom-0 left-0 right-0 bg-white text-red-600 p-4 shadow-md text-center transition-transform duration-300 ${showBanner ? "translate-y-0" : "translate-y-full"
          }`}
      >
        Uploaded images and processed results will be automatically deleted after 3 minutes. Please download your results promptly.
      </div>
    </div>
  );
}

export default App;
