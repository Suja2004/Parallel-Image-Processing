import React, { useState, useEffect } from "react";
import { Download, X, ChevronLeft, ChevronRight } from "lucide-react";
import { downloadFile, getPreviewUrl } from "./apis/api";

function OutputFiles({ task }) {
  const [previewIndex, setPreviewIndex] = useState(null);

  const openPreview = (index) => setPreviewIndex(index);
  const closePreview = () => setPreviewIndex(null);

  const showPrev = (e) => {
    e?.stopPropagation();
    setPreviewIndex((prev) =>
      prev > 0 ? prev - 1 : task.output_files.length - 1
    );
  };

  const showNext = (e) => {
    e?.stopPropagation();
    setPreviewIndex((prev) =>
      prev < task.output_files.length - 1 ? prev + 1 : 0
    );
  };

  // ⌨️ Keyboard navigation
  useEffect(() => {
    if (previewIndex === null) return;
    const handleKeyDown = (e) => {
      if (e.key === "ArrowLeft") showPrev();
      else if (e.key === "ArrowRight") showNext();
      else if (e.key === "Escape") closePreview();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [previewIndex]);

  const getProcessName = (filename) => {
    const parts = filename.split("_");
    return parts.length > 1 ? parts[parts.length - 1] : filename;
  };

  return (
    <div>
      {task.output_files?.length > 0 && (
        <>
          <h4 className="font-medium mb-2">
            Output Files ({task.output_files.length})
          </h4>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {task.output_files.map((filename, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-secondary-50 hover:bg-secondary-100 rounded transition-colors"
              >
                <span
                  className="text-sm truncate cursor-pointer text-blue-600 hover:underline"
                  onClick={() => openPreview(index)}
                  title="Preview"
                >
                  {getProcessName(filename)}
                </span>
                <button onClick={() => downloadFile(filename)}>
                  <Download className="w-4 h-4 text-primary-600" />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Modal for image preview */}
      {previewIndex !== null && (
        <div
          className="fixed inset-0 bg-[#000000b8] flex items-center justify-center z-50"
          onClick={closePreview}
        >
          <div
            className="relative bg-white p-4 rounded-lg max-w-3xl max-h-[90vh] flex flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close */}
            <button
              onClick={closePreview}
              className="absolute top-2 right-2 bg-gray-200 hover:bg-gray-300 text-gray-700 p-1 rounded-full"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Prev */}
            <button
              onClick={showPrev}
              className="absolute left-2 top-1/2 -translate-y-1/2 bg-gray-200 hover:bg-gray-300 text-gray-700 p-1 rounded-full"
            >
              <ChevronLeft className="w-6 h-6" />
            </button>

            {/* Image */}
            <img
              src={getPreviewUrl(task.output_files[previewIndex])}
              alt={task.output_files[previewIndex]}
              className="w-[600px] h-[400px] object-contain rounded mb-3"
            />

            {/* Process Name + Download */}
            <div className="text-center text-lg font-medium flex items-center justify-evenly w-full">
              {getProcessName(task.output_files[previewIndex])}
              <button
                onClick={() =>
                  downloadFile(task.output_files[previewIndex])
                }
              >
                <Download className="w-4 h-4 text-primary-600 hover:text-emerald-500" />
              </button>
            </div>

            {/* Next */}
            <button
              onClick={showNext}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-gray-200 hover:bg-gray-300 text-gray-700 p-1 rounded-full"
            >
              <ChevronRight className="w-6 h-6" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default OutputFiles;
