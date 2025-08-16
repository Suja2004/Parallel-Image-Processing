import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:5000";

// ✅ Backend health check
export const checkBackendHealth = async () => {
    return axios.get(`${API_BASE_URL}/api/health`);
};

// ✅ Fetch presets
export const fetchPresets = async () => {
    return axios.get(`${API_BASE_URL}/api/presets`);
};

// ✅ Upload files
export const uploadFiles = async (files) => {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    return axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });
};

// ✅ Process images
export const processImagesApi = async (payload, isSingle) => {
    const endpoint = isSingle ? "/api/process-multi" : "/api/batch-process";
    return axios.post(`${API_BASE_URL}${endpoint}`, payload);
};

// ✅ Poll task status
export const getTaskStatus = async (taskId) => {
    return axios.get(`${API_BASE_URL}/api/status/${taskId}`);
};

// ✅ Get image URL for preview
export const getImageUrl = (filename) => {
    return `${API_BASE_URL}/api/uploads/${filename}`;
};

// api.js
export const downloadFile = (filename) => {
    window.location.href = `${API_BASE_URL}/api/download/${filename}`;
};

export const getPreviewUrl = (filename) => {
    return `${API_BASE_URL}/api/preview/${filename}`;
};


export { API_BASE_URL };
