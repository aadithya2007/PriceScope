import React from "react";

const styles = {
  "Excellent Value": "bg-emerald-100 text-emerald-800",
  "Good Value": "bg-lime-100 text-lime-800",
  "Fair Price": "bg-amber-100 text-amber-800",
  "Slightly Expensive": "bg-orange-100 text-orange-800",
  Overpriced: "bg-rose-100 text-rose-800"
};

export default function ValueBadge({ label }) {
  return <span className={`badge ${styles[label] || "bg-stone-100 text-stone-700"}`}>{label || "Unknown"}</span>;
}
