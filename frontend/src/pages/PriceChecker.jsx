import { Calculator } from "lucide-react";
import React, { useState } from "react";
import { predictPrice } from "../api/client.js";

const defaults = {
  brand: "Lenovo",
  processor_brand: "Intel",
  processor_gen: 13,
  ram_gb: 16,
  storage_type: "SSD",
  storage_gb: 512,
  gpu_vram_gb: 0,
  display_size_inch: 15.6,
  refresh_rate_hz: 60,
  weight_kg: 1.8,
  os: "Windows 11 Home",
  retailer: "flipkart"
};

export default function PriceChecker() {
  const [form, setForm] = useState(defaults);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const submit = (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    const numeric = ["processor_gen", "ram_gb", "storage_gb", "gpu_vram_gb", "display_size_inch", "refresh_rate_hz", "weight_kg"];
    const payload = { ...form };
    numeric.forEach((key) => {
      payload[key] = Number(payload[key]);
    });
    predictPrice(payload)
      .then(setResult)
      .catch(() => setError("The model could not score those specs."))
      .finally(() => setLoading(false));
  };

  return (
    <section className="space-y-5">
      <div>
        <h2 className="text-3xl font-bold text-ink">Price Checker</h2>
        <p className="mt-2 text-stone-500">Estimate a fair market price from laptop specs.</p>
      </div>

      <form onSubmit={submit} className="panel grid gap-4 p-5 md:grid-cols-3">
        {Object.entries(defaults).map(([key]) => (
          <label key={key} className="text-sm text-stone-600">
            {key.replaceAll("_", " ")}
            <input className="field mt-1" value={form[key]} onChange={(e) => update(key, e.target.value)} />
          </label>
        ))}
        <button className="flex items-center justify-center gap-2 rounded-md bg-ink px-4 py-2 font-semibold text-white md:col-span-3">
          <Calculator className="h-4 w-4" /> {loading ? "Checking..." : "Predict Fair Price"}
        </button>
      </form>

      {error && <p className="rounded-md bg-rose-50 p-4 text-rose-700">{error}</p>}
      {result && (
        <div className="panel p-6">
          <p className="text-sm font-semibold uppercase text-clay">Predicted fair price</p>
          <p className="mt-2 text-4xl font-bold text-ink">
            Rs {new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(result.predicted_fair_price)}
          </p>
          <p className="mt-3 text-stone-600">Based on similar laptops in the PriceScope dataset and the trained Stage 3 model.</p>
        </div>
      )}
    </section>
  );
}
