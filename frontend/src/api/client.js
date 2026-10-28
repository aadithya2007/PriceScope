import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001",
  timeout: 15000
});

export const getInsights = () => api.get("/api/insights/summary").then((res) => res.data);
export const getTopValue = (limit = 10) =>
  api.get("/api/insights/top-value", { params: { limit } }).then((res) => res.data);
export const getLaptops = (params) => api.get("/api/laptops", { params }).then((res) => res.data);
export const predictPrice = (payload) => api.post("/api/predict-price", payload).then((res) => res.data);
export const recommendLaptops = (payload) => api.post("/api/recommend", payload).then((res) => res.data);

export default api;
