# PriceScope Stage 2 EDA Insights

Dataset analysed: `data\processed\laptops.csv` with **464 rows**.

## Key Findings

- Median laptop price is **Rs 0.63L**, while the 90th percentile reaches **Rs 0.80L**, confirming a broad market from budget machines to premium gaming/workstation models.
- **Lenovo** has the largest listing share in this dataset, so later recommendation logic should watch for brand imbalance.
- **Apple** is the highest-median-price brand, while **Infinix** is the lowest-median-price brand in the current catalogue.
- Dedicated-GPU laptops have a median price of **Rs 0.72L** versus **Rs 0.59L** for integrated graphics, or about **1.2x** higher.
- The strongest numeric relationship with price is **spec_score** with correlation **0.69**, based on the engineered/numeric EDA fields.
- Flipkart is cheapest on comparable brand/RAM/storage/GPU groups (-1.0% vs group average), while Comparison is priciest (0.8%).
- The most common configuration is **16GB RAM + 512GB SSD + Integrated** with **15 listings**, making it an important baseline for Stage 3 recommendation logic.
- In the Rs 50k-80k band, MSI has the strongest simple spec-score-per-lakh signal among brands represented in that range.

## Notes for Stage 3

The `spec_score` used here is intentionally naive: it normalizes RAM, storage, GPU tier, and processor tier to make value-for-money outliers visible. It should be treated as an exploratory signal only, not as the final pricing or recommendation model.
