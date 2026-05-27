"""生成 Amazon 销售模拟数据集（2023 Q1-Q4）"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

N = 12000  # 总订单数

# ── 产品池 ──
categories = {
    "电子产品": [
        ("无线蓝牙耳机", 199, 89),
        ("便携式充电宝", 79, 35),
        ("USB-C 集线器", 49, 22),
        ("手机防窥膜 3片装", 15, 5),
        ("智能手表表带", 29, 12),
        ("笔记本散热支架", 59, 28),
        ("机械键盘", 129, 60),
        ("无线鼠标", 39, 18),
    ],
    "家居厨房": [
        ("不锈钢保温杯", 35, 15),
        ("硅胶厨房 utensils 套装", 29, 10),
        ("记忆棉坐垫", 45, 18),
        ("LED 护眼台灯", 55, 25),
        ("收纳盒 6件套", 39, 14),
        ("防滑衣架 20个装", 22, 8),
    ],
    "运动户外": [
        ("瑜伽垫加厚防滑", 39, 18),
        ("阻力带套装", 19, 7),
        ("运动水壶 1L", 25, 10),
        ("登山背包 40L", 89, 42),
        ("跑步腰包", 18, 6),
    ],
    "宠物用品": [
        ("宠物饮水机", 49, 22),
        ("猫抓板沙发款", 29, 10),
        ("狗狗磨牙玩具套装", 18, 6),
        ("宠物梳毛器", 15, 5),
        ("可折叠宠物笼", 79, 35),
    ],
    "办公用品": [
        ("A4 复印纸 500张", 12, 4),
        ("桌面文件架", 32, 12),
        ("白板磁吸套装", 45, 18),
        ("人体工学腕托", 25, 9),
    ],
}

# 展开产品列表
products = []
for cat, items in categories.items():
    for name, price, cost in items:
        products.append({"category": cat, "name": name, "price": price, "cost": cost})

# ── 地区 ──
regions = {
    "北美": ["US", "CA", "MX"],
    "欧洲": ["UK", "DE", "FR", "IT", "ES"],
    "亚太": ["JP", "AU", "IN", "SG"],
    "其他": ["BR", "AE", "SA"],
}
region_list = []
for region, countries in regions.items():
    for c in countries:
        region_list.append((region, c))

# ── 生成订单 ──
orders = []
start_date = datetime(2023, 1, 1)

for i in range(N):
    prod = products[np.random.randint(0, len(products))]
    region, country = region_list[np.random.randint(0, len(region_list))]

    # 模拟季节性：Q4 旺季订单更多
    month_weights = [0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.08, 0.08, 0.10, 0.12, 0.11]
    month = np.random.choice(range(1, 13), p=month_weights)
    day = np.random.randint(1, 29)
    order_date = datetime(2023, month, day) + timedelta(
        hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60)
    )

    quantity = np.random.choice([1, 1, 1, 1, 2, 2, 3], p=[0.55, 0.15, 0.10, 0.05, 0.08, 0.04, 0.03])
    unit_price = prod["price"] * np.random.uniform(0.85, 1.15)
    revenue = round(unit_price * quantity, 2)
    cost_total = round(prod["cost"] * quantity, 2)
    profit = round(revenue - cost_total, 2)

    # 退货率约 6%
    is_returned = np.random.choice([0, 1], p=[0.94, 0.06])

    orders.append({
        "order_id": f"AMZ-{2023:04d}-{i+1:06d}",
        "order_date": order_date,
        "product_name": prod["name"],
        "category": prod["category"],
        "unit_price": round(unit_price, 2),
        "quantity": quantity,
        "revenue": revenue,
        "cost": cost_total,
        "profit": profit,
        "region": region,
        "country": country,
        "is_returned": is_returned,
    })

df = pd.DataFrame(orders)

# 保存
out_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(out_dir, "amazon_sales_2023.csv")
df.to_csv(path, index=False, encoding="utf-8-sig")

print(f"已生成 {len(df)} 条订单数据 → {path}")
print(f"数据概览:\n{df.describe(include='all')}")
print(f"\n总营收: ${df['revenue'].sum():,.2f}")
print(f"总利润: ${df['profit'].sum():,.2f}")
print(f"退货率: {df['is_returned'].mean():.2%}")
