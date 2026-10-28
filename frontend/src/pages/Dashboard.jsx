import { Award, IndianRupee, Laptop } from "lucide-react";
import React, { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getInsights } from "../api/client.js";
import ValueBadge from "../components/ValueBadge.jsx";

const money = (value) => new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value || 0);

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getInsights().then(setData).catch(() => setError("Could not load dashboard insights."));
  }, []);

  if (error) return <p className="rounded-md bg-rose-50 p-4 text-rose-700">{error}</p>;
  if (!data) return <p className="text-stone-500">Loading dashboard...</p>;

  const cards = [
    { label: "Laptops tracked", value: data.total_laptops, icon: Laptop },
    { label: "Average price", value: `Rs ${money(data.average_price)}`, icon: IndianRupee },
    { label: "Best value pick", value: data.best_value_pick?.brand || "N/A", icon: Award }
  ];

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-ink">Market Dashboard</h2>
        <p className="mt-2 text-stone-500">Live summaries from the cleaned, value-labeled laptop dataset.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {cards.map(({ label, value, icon: Icon }) => (
          <div key={label} className="panel p-5">
            <Icon className="h-5 w-5 text-clay" />
            <p className="mt-4 text-sm text-stone-500">{label}</p>
            <p className="mt-1 text-2xl font-semibold text-ink">{value}</p>
          </div>
        ))}
      </div>

      {data.best_value_pick && (
        <div className="panel flex flex-col gap-3 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase text-clay">Pick of the day</p>
            <h3 className="mt-1 text-lg font-semibold text-ink">{data.best_value_pick.model_name}</h3>
            <p className="text-sm text-stone-500">
              Rs {money(data.best_value_pick.price_inr)} actual vs Rs {money(data.best_value_pick.predicted_fair_price)} fair
            </p>
          </div>
          <ValueBadge label={data.best_value_pick.value_label} />
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="panel p-5">
          <h3 className="font-semibold text-ink">Price Distribution</h3>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.price_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#445f48" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="panel p-5">
          <h3 className="font-semibold text-ink">Brand Average Price</h3>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.brand_averages}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="brand" />
                <YAxis />
                <Tooltip formatter={(value) => `Rs ${money(value)}`} />
                <Bar dataKey="avg_price" fill="#b86d47" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
