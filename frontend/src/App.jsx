import { BarChart3, Gauge, Laptop, Search } from "lucide-react";
import React, { useState } from "react";
import Browse from "./pages/Browse.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import PriceChecker from "./pages/PriceChecker.jsx";
import Recommend from "./pages/Recommend.jsx";

const tabs = [
  { id: "dashboard", label: "Dashboard", icon: BarChart3, component: Dashboard },
  { id: "browse", label: "Browse", icon: Laptop, component: Browse },
  { id: "checker", label: "Price Checker", icon: Gauge, component: PriceChecker },
  { id: "recommend", label: "Recommend", icon: Search, component: Recommend }
];

export default function App() {
  const [active, setActive] = useState("dashboard");
  const ActivePage = tabs.find((tab) => tab.id === active)?.component || Dashboard;

  return (
    <div className="min-h-screen bg-paper">
      <aside className="fixed inset-y-0 left-0 hidden w-64 border-r border-stone-200 bg-white px-4 py-6 lg:block">
        <h1 className="text-2xl font-bold text-ink">PriceScope</h1>
        <p className="mt-2 text-sm text-stone-500">Laptop value intelligence</p>
        <nav className="mt-8 space-y-2">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActive(id)}
              className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm font-medium transition ${
                active === id ? "bg-mint text-ink" : "text-stone-600 hover:bg-stone-100"
              }`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </nav>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-10 border-b border-stone-200 bg-white/95 px-4 py-3 backdrop-blur lg:hidden">
          <div className="flex flex-wrap gap-2">
            {tabs.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setActive(id)}
                className={`rounded-md px-3 py-2 text-sm font-medium ${active === id ? "bg-mint text-ink" : "text-stone-600"}`}
              >
                {label}
              </button>
            ))}
          </div>
        </header>
        <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <ActivePage />
        </main>
      </div>
    </div>
  );
}
