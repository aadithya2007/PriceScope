/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17211c",
        moss: "#445f48",
        mint: "#d9efe1",
        clay: "#b86d47",
        amber: "#f2b84b",
        paper: "#f8faf7"
      },
      boxShadow: {
        soft: "0 12px 30px rgba(23, 33, 28, 0.08)"
      }
    }
  },
  plugins: []
};

