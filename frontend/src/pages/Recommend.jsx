import { Search } from "lucide-react";
import React, { useState } from "react";
import { recommendLaptops } from "../api/client.js";
import LaptopCard from "../components/LaptopCard.jsx";

export default function Recommend() {
  const [form, setForm] = useState({ budget: 70000, ram_min: 16, storage_min: 512, usage_type: "programming" });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    recommendLaptops({
      ...form,
      budget: Number(form.budget),
      ram_min: Number(form.ram_min),
      storage_min: Number(form.storage_min)
    })
      .then(setResults)
      .catch(() => setError("Recommendations could not be generated."))
      .finally(() => setLoading(false));
  };

  return (
    <section className="space-y-5">
      <div>
        <h2 className="text-3xl font-bold text-ink">Recommend For Me</h2>
        <p className="mt-2 text-stone-500">Ranked matches combine workload fit with fair-price value.</p>
      </div>

      <form onSubmit={submit} className="panel grid gap-4 p-5 md:grid-cols-5">
        <input className="field" type="number" value={form.budget} onChange={(e) => setForm({ ...form, budget: e.target.value })} placeholder="Budget" />
        <input className="field" type="number" value={form.ram_min} onChange={(e) => setForm({ ...form, ram_min: e.target.value })} placeholder="Min RAM" />
        <input className="field" type="number" value={form.storage_min} onChange={(e) => setForm({ ...form, storage_min: e.target.value })} placeholder="Min storage" />
        <select className="field" value={form.usage_type} onChange={(e) => setForm({ ...form, usage_type: e.target.value })}>
          <option value="gaming">Gaming</option>
          <option value="programming">Programming</option>
          <option value="student">Student</option>
          <option value="general">General</option>
        </select>
        <button className="flex items-center justify-center gap-2 rounded-md bg-ink px-4 py-2 font-semibold text-white">
          <Search className="h-4 w-4" /> {loading ? "Finding..." : "Find"}
        </button>
      </form>

      {error && <p className="rounded-md bg-rose-50 p-4 text-rose-700">{error}</p>}
      {!loading && results.length === 0 && <p className="rounded-md bg-white p-4 text-stone-500">Submit your preferences to see ranked matches.</p>}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {results.map((item, index) => (
          <LaptopCard
            key={item.laptop.laptop_id}
            laptop={item.laptop}
            rank={index + 1}
            score={item.match_score}
            why={item.why_recommended}
          />
        ))}
      </div>
    </section>
  );
}
