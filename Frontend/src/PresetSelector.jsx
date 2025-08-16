import { useState } from "react";
import { HelpCircle, X } from "lucide-react";

/** High-level preset descriptions */
const PRESET_DESCRIPTIONS = {
    edge_detection_suite: {
        summary:
            "Highlights object boundaries using multiple edge finders at different sensitivities.",
        expected:
            "Crisp white lines on dark background; thin to slightly thicker edges depending on the setting.",
    },
    blur_variations: {
        summary:
            "Progressively smooths the image to reduce noise or soften details.",
        expected:
            "From gentle softening to strong blur; median filter removes speckle-like noise while preserving edges better.",
    },
    enhancement_pack: {
        summary:
            "Boosts clarity by sharpening edges, then tames noise with gentle smoothing.",
        expected:
            "Sharper details with fewer harsh artifacts, good balance between clarity and smoothness.",
    },
    comparison_set: {
        summary:
            "A mixed bag to compare edge detection, smoothing, and sharpening side by side.",
        expected:
            "One edge image, one smoothed image, and one sharpened image to help you choose.",
    },
};

/** Operation descriptions */
const OPERATION_INFO = {
    canny: {
        label: "Canny Edge Detection",
        what:
            "Finds strong, clean edges by looking for rapid changes in brightness and linking them.",
        output: "Binary edges (mostly black background with white lines).",
    },
    sobel: {
        label: "Sobel Edge Map",
        what:
            "Measures gradient strength to show where image intensity changes; softer than Canny.",
        output: "Grayscale edge magnitude (brighter = stronger edge).",
    },
    gaussian: {
        label: "Gaussian Blur",
        what:
            "Evenly smooths the image by averaging pixels with a bell-shaped weight.",
        output: "Smoothed image, details gradually softened.",
    },
    median: {
        label: "Median Filter",
        what:
            "Replaces each pixel with the median of its neighbors; great for salt-and-pepper noise.",
        output: "Cleaner image with preserved edges; speckle noise reduced.",
    },
    sharpen: {
        label: "Sharpen",
        what:
            "Emphasizes edges by boosting high-frequency detail using an unsharp-like kernel.",
        output: "Sharper details; may increase noise if used aggressively.",
    },
};

/** Nicely format parameters */
const PARAM_LABELS = { t1: "Low threshold", t2: "High threshold", ksize: "Kernel size", sigma: "Sigma" };
function formatParams(params = {}) {
    const entries = Object.entries(params);
    if (!entries.length) return "None";
    return entries.map(([k, v]) => `${PARAM_LABELS[k] || k}: ${v}`).join(", ");
}

function PresetSelector({ presets, selectedPreset, setSelectedPreset }) {
    const [showModal, setShowModal] = useState(false);

    return (
        <div className="mb-4">
            <label className="block text-sm font-medium text-secondary-700 mb-2">
                Select Filter Preset
            </label>

            <div className="flex items-center gap-2">
                <select
                    value={selectedPreset}
                    onChange={(e) => setSelectedPreset(e.target.value)}
                    className="bg-white border border-gray-300 rounded p-2 text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-400 flex-1 outline-0 "
                >
                    <option value="">Choose a preset...</option>
                    {Object.entries(presets).map(([key, filters]) => (
                        <option key={key} value={key}>
                            {key.replace("_", " ").toUpperCase()} ({filters.length} filters)
                        </option>
                    ))}
                </select>

                {/* Single static help button */}
                <button
                    type="button"
                    className="p-2 rounded-full bg-gray-100 hover:bg-gray-200"
                    onClick={() => setShowModal(true)}
                    aria-label="Show all preset details"
                    title="Show all preset and operation details"
                >
                    <HelpCircle className="w-5 h-5 text-gray-600" />
                </button>
            </div>

            {selectedPreset && presets[selectedPreset] && (
                <div className="mt-3 p-3 bg-secondary-50 rounded-lg">
                    <h4 className="font-medium text-secondary-700 mb-2">
                        Preset Details:
                    </h4>
                    <div className="space-y-1">
                        {presets[selectedPreset].map((filter, index) => (
                            <div key={index} className="bg-white rounded-lg shadow-lg max-w-md w-full p-3">
                                • {filter.name}: {filter.operation}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Modal showing all presets */}
            {showModal && (
                <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setShowModal(false)}
                >
                    <div className="bg-white rounded-lg shadow-lg max-w-3xl w-full p-6 relative overflow-y-auto max-h-[80vh]">
                        <button
                            onClick={() => setShowModal(false)}
                            className="absolute top-2 right-2 p-1 rounded-full bg-gray-100 hover:bg-gray-200"
                        >
                            <X className="w-5 h-5 text-gray-600  right-10" />
                        </button>

                        <h3 className="text-lg font-semibold mb-4 ">All Presets & Operations</h3>

                        {Object.entries(presets).map(([presetKey, filters]) => (
                            <div key={presetKey} className="mb-6">
                                <h4 className="font-medium text-secondary-700 text-sm mb-1">
                                    {presetKey.replace("_", " ").toUpperCase()}
                                </h4>
                                <p className="text-xs text-secondary-600 mb-2">
                                    {PRESET_DESCRIPTIONS[presetKey]?.summary}
                                </p>
                                {PRESET_DESCRIPTIONS[presetKey]?.expected && (
                                    <p className="text-xs text-gray-500 mb-2">
                                        Expected look: {PRESET_DESCRIPTIONS[presetKey].expected}
                                    </p>
                                )}

                                <div className="space-y-2">
                                    {filters.map((filter, idx) => {
                                        const op = OPERATION_INFO[filter.operation] || {};
                                        return (
                                            <div key={idx} className="p-3 border rounded-lg bg-gray-50 text-sm">
                                                <div className="font-medium">
                                                    {filter.name}{" "}
                                                    <span className="text-gray-500">
                                                        ({op.label || filter.operation})
                                                    </span>
                                                </div>
                                                {op.what && (
                                                    <div className="mt-1 text-gray-700">
                                                        <span className="font-medium">What it does:</span> {op.what}
                                                    </div>
                                                )}
                                                {op.output && (
                                                    <div className="mt-1 text-gray-700">
                                                        <span className="font-medium">Output:</span> {op.output}
                                                    </div>
                                                )}
                                                {filter.parameters && Object.keys(filter.parameters).length > 0 && (
                                                    <div className="mt-1 text-gray-700">
                                                        <span className="font-medium">Parameters:</span>{" "}
                                                        {formatParams(filter.parameters)}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default PresetSelector;
