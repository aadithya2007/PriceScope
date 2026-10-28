import { IndianRupee, Star } from "lucide-react";
import React from "react";
import ValueBadge from "./ValueBadge.jsx";

const money = (value) =>
  value == null ? "N/A" : new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value);

export default function LaptopCard({ laptop, rank, score, why }) {
  return (
    <article className="panel flex h-full flex-col gap-4 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          {rank && <p className="text-xs font-semibold uppercase tracking-wide text-clay">Rank {rank}</p>}
          <h3 className="mt-1 line-clamp-2 text-base font-semibold text-ink">{laptop.model_name || "Laptop"}</h3>
          <p className="mt-1 text-sm text-stone-500">
            {laptop.brand || "Unknown"} · {laptop.retailer || "retailer"}
          </p>
        </div>
        <ValueBadge label={laptop.value_label} />
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md bg-paper p-3">
          <p className="text-stone-500">Actual</p>
          <p className="mt-1 flex items-center font-semibold text-ink">
            <IndianRupee className="h-4 w-4" /> {money(laptop.price_inr || laptop.actual_price)}
          </p>
        </div>
        <div className="rounded-md bg-paper p-3">
          <p className="text-stone-500">Fair price</p>
          <p className="mt-1 flex items-center font-semibold text-ink">
            <IndianRupee className="h-4 w-4" /> {money(laptop.predicted_fair_price)}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs text-stone-600">
        <span>{laptop.ram_gb || "-"}GB RAM</span>
        <span>{laptop.storage_gb || "-"}GB</span>
        <span>{laptop.gpu_vram_gb || 0}GB GPU</span>
      </div>

      {score != null && (
        <div>
          <div className="mb-1 flex justify-between text-xs text-stone-500">
            <span>Match score</span>
            <span>{score}%</span>
          </div>
          <div className="h-2 rounded-full bg-stone-100">
            <div className="h-2 rounded-full bg-moss" style={{ width: `${Math.min(score, 100)}%` }} />
          </div>
        </div>
      )}

      {why && <p className="mt-auto text-sm text-stone-600">{why}</p>}
      {laptop.rating && (
        <p className="flex items-center gap-1 text-xs text-stone-500">
          <Star className="h-3.5 w-3.5 fill-amber text-amber" /> {laptop.rating}
        </p>
      )}
    </article>
  );
}
