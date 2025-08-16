import React, { useState, useEffect } from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { getImageUrl } from "./apis/api";

function ImageModal({ files, }) {
  const [currentIndex, setCurrentIndex] = useState(null);

  // === Handlers ===
  const openModal = (index) => setCurrentIndex(index);
  const closeModal = () => setCurrentIndex(null);

  const showPrev = (e) => {
    if (e) e.stopPropagation();
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : files.length - 1));
  };

  const showNext = (e) => {
    if (e) e.stopPropagation();
    setCurrentIndex((prev) => (prev < files.length - 1 ? prev + 1 : 0));
  };

  // === Keyboard Controls ===
  useEffect(() => {
    if (currentIndex === null) return;
    const handleKeys = (e) => {
      if (e.key === "Escape") closeModal();
      if (e.key === "ArrowLeft") showPrev();
      if (e.key === "ArrowRight") showNext();
    };
    window.addEventListener("keydown", handleKeys);
    return () => window.removeEventListener("keydown", handleKeys);
  }, [currentIndex]);

  if (files.length === 0) return null;

  return (
    <div className="mt-4">
      <h3 className="text-lg font-medium mb-2">
        Uploaded Files ({files.length})
      </h3>

      {/* Thumbnails */}
      <div className="space-y-2 max-h-40 overflow-y-auto">
        {files.map((file, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-2 bg-secondary-50 rounded"
          >
            <img
              src={getImageUrl(file.filename)}
              alt={file.original_filename}
              className="w-16 h-16 object-contain rounded cursor-pointer hover:opacity-80 transition"
              onClick={() => openModal(index)}
            />
            <span className="text-sm text-secondary-700 truncate">
              {file.original_filename}
            </span>
            <span className="text-xs text-secondary-500">
              {file.dimensions.width}×{file.dimensions.height}
            </span>
          </div>
        ))}
      </div>

      {/* Modal */}
      {currentIndex !== null && (
        <div
          className="fixed inset-0 bg-[#000000b8] flex items-center justify-center z-50"
          onClick={closeModal}
        >
          <div
            className="relative bg-white p-2 rounded-lg max-w-4xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close */}
            <button
              onClick={closeModal}
              className="absolute top-2 right-2 bg-gray-200 hover:bg-gray-300 text-gray-700 p-1 rounded-full"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Prev */}
            <button
              onClick={showPrev}
              className="absolute left-2 top-1/2 -translate-y-1/2 bg-gray-200 hover:bg-gray-300 text-gray-700 p-2 rounded-full"
            >
              <ChevronLeft className="w-6 h-6" />
            </button>

            {/* Next */}
            <button
              onClick={showNext}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-gray-200 hover:bg-gray-300 text-gray-700 p-2 rounded-full"
            >
              <ChevronRight className="w-6 h-6" />
            </button>

            {/* Full Image */}
            <img
              src={getImageUrl(files[currentIndex].filename)}
              alt={files[currentIndex].original_filename}
              className="w-[600px] h-[400px] object-contain rounded"
            />

            {/* Caption */}
            <div className="text-center mt-2 text-gray-600 text-sm">
              {files[currentIndex].original_filename} (
              {files[currentIndex].dimensions?.width}×
              {files[currentIndex].dimensions?.height})
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageModal;
