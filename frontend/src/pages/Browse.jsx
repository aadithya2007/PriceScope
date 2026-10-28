import { SlidersHorizontal } from "lucide-react";
import React, { useEffect, useState } from "react";
import { getLaptops } from "../api/client.js";
import LaptopCard from "../components/LaptopCard.jsx";

const initialFilters = { brand: "", retailer: "", max_price: "", min_ram: "", value_label: "" };

export default function Browse() {
  const [filters, setFilters] = useState(initialFilters);
  const [laptops, setLaptops] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = () => {
    setLoading(true);
    setError("");
    const params = Object.fromEntries(Object.entries(filters).filter(([, value]) => value !== ""));
    getLaptops({ ...params, page_size: 24 })
      .then((res) => {
        setLaptops(res.items);
        setTotal(res.total);
      })
      .catch(() => setError("No listings could be loaded."))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  return (
    <section className="space-y-5">
      <div>
        <h2 className="text-3xl font-bold text-ink">Browse Laptops</h2>
        <p className="mt-2 text-stone-500">Filter listings by value, retailer, price, and minimum specs.</p>
      </div>

      <div className="panel grid gap-3 p-4 md:grid-cols-6">
        <input className="field" placeholder="Brand" value={filters.brand} onChange={(e) => setFilters({ ...filters, brand: e.target.value })} />
        <select className="field" value={filters.retailer} onChange={(e) => setFilters({ ...filters, retailer: e.target.value })}>
          <option value="">Retailer</option>
          <option value="amazon">Amazon</option>
          <option value="flipkart">Flipkart</option>
          <option value="croma">Croma</option>
          <option value="comparison">Comparison</option>
        </select>
        <input className="field" type="number" placeholder="Max price" value={filters.max_price} onChange={(e) => setFilters({ ...filters, max_price: e.target.value })} />
        <input className="field" type="number" placeholder="Min RAM" value={filters.min_ram} onChange={(e) => setFilters({ ...filters, min_ram: e.target.value })} />
        <select className="field" value={filters.value_label} onChange={(e) => setFilters({ ...filters, value_label: e.target.value })}>
          <option value="">Value label</option>
          <option>Excellent Value</option>
          <option>Good Value</option>
          <option>Fair Price</option>
          <option>Slightly Expensive</option>
          <option>Overpriced</option>
        </select>
        <button onClick={load} className="flex items-center justify-center gap-2 rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white">
          <SlidersHorizontal className="h-4 w-4" /> Apply
        </button>
      </div>

      <p className="text-sm text-stone-500">{loading ? "Loading..." : `${total} listings found`}</p>
      {error && <p className="rounded-md bg-rose-50 p-4 text-rose-700">{error}</p>}
      {!loading && laptops.length === 0 && <p className="rounded-md bg-white p-4 text-stone-500">No laptops match those filters.</p>}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {laptops.map((laptop) => (
          <LaptopCard key={laptop.laptop_id} laptop={laptop} />
        ))}
      </div>
    </section>
  );
}
